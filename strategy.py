#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_HTFVolume_v1
Hypothesis: Camarilla R1/S1 breakout on 12h with 1-week EMA50 trend filter and 1-week volume spike (>2.0x median). Targets institutional weekly pivot levels with strong volume confirmation in weekly trending markets. Designed for BTC/ETH with strict entry conditions (~12-30 trades/year) to avoid fee drift. Uses discrete position sizing (0.30) and ATR trailing stop (2.5x) for risk control. Works in bull/bear by only trading with weekly trend direction.
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
    
    # Get 1w data for HTF trend and volume
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1w volume median (30-period) for spike filter
    vol_median_1w = pd.Series(df_1w['volume'].values).rolling(window=30, min_periods=30).median().values
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Camarilla levels from previous 12h bar (HLC of prior 12h)
    cam_high = pd.Series(df_12h['high'].values).shift(1).values
    cam_low = pd.Series(df_12h['low'].values).shift(1).values
    cam_close = pd.Series(df_12h['close'].values).shift(1).values
    
    # Camarilla R1, S1 levels (core breakout levels)
    R1 = cam_close + (cam_high - cam_low) * 1.1 / 12
    S1 = cam_close - (cam_high - cam_low) * 1.1 / 12
    
    # ATR(20) for volatility-based stops (using 12h data)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF indicators to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_median_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_median_1w)
    R1_aligned = align_htf_to_ltf(prices, df_12h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_12h, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(50) 1w, volume median (30), Camarilla (need 2 bars for shift), ATR (20)
    start_idx = max(50, 30, 2, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_median_1w_aligned[i]) or
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        vol_median_1w_val = vol_median_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        atr_val = atr[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        
        # Trend filter: price > EMA50 (weekly uptrend) or < EMA50 (weekly downtrend)
        uptrend = close_val > ema_50_1w_val
        downtrend = close_val < ema_50_1w_val
        
        # Volume spike filter: only trade in high-volume environments (weekly)
        volume_spike = volume_val > 2.0 * vol_median_1w_val
        
        if position == 0:
            # Long: break above R1 with volume spike, and weekly uptrend
            long_signal = (close_val > r1_val) and \
                          volume_spike and \
                          uptrend
            
            # Short: break below S1 with volume spike, and weekly downtrend
            short_signal = (close_val < s1_val) and \
                           volume_spike and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop
            if close_val < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop
            if close_val > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrend_HTFVolume_v1"
timeframe = "12h"
leverage = 1.0