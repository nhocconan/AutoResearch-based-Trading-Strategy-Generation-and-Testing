#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX trend filter
# - Long: price breaks above Donchian upper band (20-period high) with volume > 1.5x average and ADX > 25
# - Short: price breaks below Donchian lower band (20-period low) with volume > 1.5x average and ADX > 25
# - Exit: price returns to Donchian midpoint (mean reversion at channel center)
# - Uses 1d EMA(50) for higher timeframe trend filter: only long when price > EMA50, short when price < EMA50
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Proven pattern: Donchian breakouts with volume and trend filters work on SOLUSDT (test Sharpe 1.10-1.38)

name = "4h_donchian_breakout_volume_adx_v1"
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Pre-compute 4h ADX (14-period) for trend strength
    # ADX calculation: +DM, -DM, TR, then smoothed
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_series.diff()
    down_move = low_series.diff().multiply(-1)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean()
    atr_smooth = atr.ewm(alpha=1/14, adjust=False).mean()
    
    # DI and DX
    plus_di = 100 * (plus_dm_smooth / atr_smooth)
    minus_di = 100 * (minus_dm_smooth / atr_smooth)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    
    adx_values = adx.values
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(adx_values[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Donchian levels
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        mid_band = donchian_mid[i]
        
        # Trend filter: price relative to 1d EMA50
        price_above_ema = close_price > ema_50_aligned[i]
        price_below_ema = close_price < ema_50_aligned[i]
        
        # Trend strength: ADX > 25 indicates strong trend
        strong_trend = adx_values[i] > 25
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above upper band with volume, trend filter, and strong trend
        if close_price > upper_band and vol_confirm and price_above_ema and strong_trend:
            enter_long = True
        
        # Short breakout: price breaks below lower band with volume, trend filter, and strong trend
        if close_price < lower_band and vol_confirm and price_below_ema and strong_trend:
            enter_short = True
        
        # Exit conditions: mean reversion at Donchian midpoint
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to midpoint or below
            exit_long = close_price <= mid_band
        elif position == -1:
            # Exit short if price returns to midpoint or above
            exit_short = close_price >= mid_band
        
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