#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike_ATRStop_v3
Hypothesis: 12-hour Camarilla R1/S1 breakout with 1-week EMA50 trend filter and volume spike confirmation.
Uses 12h primary with 1w HTF for trend alignment to reduce noise and improve signal quality.
Targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
In bull: longs on breakouts above R1 with 1w uptrend. In bear: shorts on breakdowns below S1 with 1w downtrend.
Volume spike (>2.5x 20-bar MA) confirms conviction. ATR-based stoploss (2.0x ATR) controls downside.
Discrete sizing (0.0, ±0.30) balances profit potential with risk control.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss on 12h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume ratio (current / 20-period average) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)  # avoid division by zero
    
    # Calculate Camarilla levels from previous 1d bar (using 1d for pivot calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1w EMA(50), ATR(14), volume MA(20), 1d data availability
    start_idx = max(50, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ratio[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        close_val = close[i]
        vol_spike = vol_ratio[i] > 2.5  # volume at least 2.5x average (stricter)
        trend_1w_up = close_val > ema_50_1w_aligned[i]
        trend_1w_down = close_val < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND 1w trend up AND volume spike
            long_signal = (close_val > camarilla_r1_aligned[i]) and trend_1w_up and vol_spike
            
            # Short: price breaks below Camarilla S1 AND 1w trend down AND volume spike
            short_signal = (close_val < camarilla_s1_aligned[i]) and trend_1w_down and vol_spike
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: trend flips down OR price hits ATR stoploss
            if (not trend_1w_up) or (close_val < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: trend flips up OR price hits ATR stoploss
            if (not trend_1w_down) or (close_val > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike_ATRStop_v3"
timeframe = "12h"
leverage = 1.0