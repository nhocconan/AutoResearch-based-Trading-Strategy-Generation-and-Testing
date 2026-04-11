#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with volume confirmation and ADX trend filter
# - Long: price breaks above Donchian upper (20), volume > 1.5x 20-period avg, ADX(14) > 25 (strong trend)
# - Short: price breaks below Donchian lower (20), volume > 1.5x 20-period avg, ADX(14) > 25 (strong trend)
# - Exit: price returns to Donchian midpoint (mean of upper and lower)
# - Uses 1d EMA(50) trend filter: price > EMA for long bias, price < EMA for short bias
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-25 trades/year (50-100 total over 4 years) to stay within fee drag limits
# - Donchian channels work well in both trending and ranging markets with proper filters

name = "12h_1d_donchian_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for EMA trend filter and Donchian calculation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1d OHLC for Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels for each 1d bar
    # Upper = max(high, 20)
    # Lower = min(low, 20)
    # Middle = (Upper + Lower) / 2
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2
    
    # Align Donchian levels to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_1d, middle_20)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ADX for trend strength
    # True Range
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    # +DI and -DI
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / tr_14
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or np.isnan(middle_20_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(adx[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Donchian levels
        upper_level = upper_20_aligned[i]
        lower_level = lower_20_aligned[i]
        middle_level = middle_20_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Trend filter: ADX > 25 (indicates strong trend)
        strong_trend = adx[i] > 25
        
        # 1d EMA trend bias
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above Donchian upper, volume confirmation, strong trend, long bias
        if close_price > upper_level and vol_confirm and strong_trend and ema_bias_long:
            enter_long = True
        
        # Short breakout: price below Donchian lower, volume confirmation, strong trend, short bias
        if close_price < lower_level and vol_confirm and strong_trend and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to Donchian midpoint
            exit_long = close_price <= middle_level
        elif position == -1:
            # Exit short if price returns to Donchian midpoint
            exit_short = close_price >= middle_level
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals