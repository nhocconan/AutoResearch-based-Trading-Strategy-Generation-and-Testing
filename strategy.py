#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike_v1
Hypothesis: Daily Donchian(20) breakout with 1-week EMA34 trend filter and volume confirmation (>2x median) to capture strong momentum moves. Works in bull/bear markets by only trading with the weekly trend. Targets 7-25 trades/year via tight entry conditions requiring confluence of breakout, trend, and volume. Uses ATR(14) trailing stop for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend (EMA34)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data for Donchian calculation (from previous daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian levels from previous 20 daily OHLC
    prev_high_20d = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    prev_low_20d = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align HTF indicators to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    prev_high_20d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_20d)
    prev_low_20d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_20d)
    
    # Volume spike filter: volume > 2x median volume (20-period) for conviction
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR(14) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(34) 1w, Donchian(20), volume median (20), ATR (14)
    start_idx = max(34, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_median[i]) or
            np.isnan(prev_high_20d_aligned[i]) or
            np.isnan(prev_low_20d_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_34_1w_val = ema_34_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        
        # Trend filter: price > EMA34 (uptrend) or < EMA34 (downtrend)
        uptrend = close_val > ema_34_1w_val
        downtrend = close_val < ema_34_1w_val
        
        # Volume spike filter: only trade in above-average volume environments
        volume_spike = volume_val > 2.0 * vol_median_val
        
        if position == 0:
            # Long: break above upper Donchian with volume spike, and uptrend
            long_signal = (close_val > prev_high_20d_aligned[i]) and \
                          volume_spike and \
                          uptrend
            
            # Short: break below lower Donchian with volume spike, and downtrend
            short_signal = (close_val < prev_low_20d_aligned[i]) and \
                           volume_spike and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop
            if close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop
            if close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0