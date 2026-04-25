#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_VolumeSpike_ATR_Stop
Hypothesis: Daily Donchian(20) breakout with volume spike (>2.0x 20-day average) and ATR(14) stoploss (2.0*ATR). Uses discrete sizing (0.25) and holds until stoploss hit or opposite breakout. Designed to capture strong momentum bursts in both bull and bear markets by requiring high-volume confirmation. Targets 7-25 trades/year on 1d to minimize fee drag while maintaining edge in volatile regimes. Uses 1w trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ATR - primary timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 1d using previous bar's data to avoid look-ahead
    # Upper = max(high of last 20 bars), Lower = min(low of last 20 bars)
    # We shift by 1 to use only completed bars
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Calculate ATR(14) for stoploss on 1d
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter - higher timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        # Fallback to EMA50 on 1d if 1w data insufficient
        ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
        trend_filter = close_1d > ema_50_aligned  # Uptrend filter
    else:
        close_1w = df_1w['close'].values
        # Calculate 50-period EMA on 1w for trend filter
        ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
        trend_filter = close_1d > ema_50_1w_aligned  # Uptrend filter based on 1w EMA50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(20, 20, 14)  # Donchian needs 20, vol needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        upper_val = upper_20_aligned[i]
        lower_val = lower_20_aligned[i]
        atr_val = atr_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Donchian breakout with volume confirmation AND trend filter
            # Only take longs in uptrend, shorts in downtrend
            is_uptrend = trend_filter[i]
            
            # Long: price breaks above upper Donchian channel with volume spike AND uptrend
            long_signal = (high_val > upper_val) and volume_spike and is_uptrend
            # Short: price breaks below lower Donchian channel with volume spike AND downtrend
            short_signal = (low_val < lower_val) and volume_spike and (not is_uptrend)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Opposite breakout: price breaks below lower Donchian (exit long)
            elif close_val < lower_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Opposite breakout: price breaks above upper Donchian (exit short)
            elif close_val > upper_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "1d_Donchian20_Breakout_VolumeSpike_ATR_Stop"
timeframe = "1d"
leverage = 1.0