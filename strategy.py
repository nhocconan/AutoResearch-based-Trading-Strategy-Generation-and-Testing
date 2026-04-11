#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR trend filter
# - Long: price breaks above Donchian upper band (20-period high), volume > 1.5x 20-period avg, ATR(14) > ATR(50) (trending)
# - Short: price breaks below Donchian lower band (20-period low), volume > 1.5x 20-period avg, ATR(14) > ATR(50) (trending)
# - Exit: price returns to Donchian midpoint (mean of upper/lower band) or opposite band touch
# - Uses 12h EMA(50) trend filter: price > EMA for long bias, price < EMA for short bias
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-30 trades/year (80-120 total over 4 years) to stay within fee drag limits
# - Donchian channels work well in both trending and ranging markets with proper filters

name = "4h_12h_donchian_atr_volume_v1"
timeframe = "4h"
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
    
    # Load 12h data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Pre-compute Donchian channels (20-period) on 4h data
    high_roll_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_band = high_roll_20
    lower_band = low_roll_20
    middle_band = (upper_band + lower_band) / 2
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR filters for regime detection
    # True Range
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    # ATR(14) for current volatility
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # ATR(50) for longer-term volatility comparison
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(middle_band[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Donchian levels
        upper_level = upper_band[i]
        lower_level = lower_band[i]
        middle_level = middle_band[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Trend filter: ATR(14) > ATR(50) (indicates trending market)
        atr_trend = atr_14[i] > atr_50[i]
        
        # 12h EMA trend bias
        ema_bias_long = close_price > ema_50_12h_aligned[i]
        ema_bias_short = close_price < ema_50_12h_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above Donchian upper band, volume confirmation, trending, long bias
        if close_price > upper_level and vol_confirm and atr_trend and ema_bias_long:
            enter_long = True
        
        # Short breakout: price below Donchian lower band, volume confirmation, trending, short bias
        if close_price < lower_level and vol_confirm and atr_trend and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to middle band or touches lower band
            exit_long = close_price <= middle_level
        elif position == -1:
            # Exit short if price returns to middle band or touches upper band
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