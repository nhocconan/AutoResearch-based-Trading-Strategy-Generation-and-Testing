#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Pivot_Breakout_1dTrend_VolumeSpike
Hypothesis: Trade Camarilla R1/S1 breakouts in the direction of the 1d EMA34 trend with volume confirmation.
Works in both bull and bear markets by aligning with the daily trend while using intraday precision for entries.
Volume spike filter ensures participation, reducing false breakouts. Designed for 4h timeframe with discrete 0.25 position size.
Target: 20-40 trades/year per symbol for optimal generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_raw = df_1d['close'].values
    
    # Camarilla R1, S1, R3, S3 from previous day
    R1 = close_1d_raw + 1.1 * (high_1d - low_1d) / 12
    S1 = close_1d_raw - 1.1 * (high_1d - low_1d) / 12
    R3 = close_1d_raw + 1.1 * (high_1d - low_1d) / 4
    S3 = close_1d_raw - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels (use previous day's levels for current bar)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1, additional_delay_bars=1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1, additional_delay_bars=1)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3, additional_delay_bars=1)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3, additional_delay_bars=1)
    
    # 4h ATR(20) for volume spike and stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    vol = prices['volume'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Volume spike: current volume > 2.0 * average volume of last 20 bars
    vol_ma = pd.Series(vol).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    fixed_size = 0.25
    
    # Warmup: need 34 for 1d EMA, 20 for ATR/volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for breakout in trend direction with volume confirmation
            if vol_spike:
                # 1d trend: price above/below EMA34
                uptrend = close_val > ema_34_1d_aligned[i]
                downtrend = close_val < ema_34_1d_aligned[i]
                
                # Long: price breaks above R1 with volume in uptrend
                long_entry = uptrend and (close_val > R1_aligned[i])
                # Short: price breaks below S1 with volume in downtrend
                short_entry = downtrend and (close_val < S1_aligned[i])
                
                if long_entry:
                    signals[i] = fixed_size
                    position = 1
                elif short_entry:
                    signals[i] = -fixed_size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reaches R3 (profit target) or breaks below S1 (stop)
            if close_val >= R3_aligned[i] or close_val < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = fixed_size
        elif position == -1:
            # Short - exit when price reaches S3 (profit target) or breaks above R1 (stop)
            if close_val <= S3_aligned[i] or close_val > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -fixed_size
    
    return signals

name = "4h_Camarilla_R1S1_Pivot_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0