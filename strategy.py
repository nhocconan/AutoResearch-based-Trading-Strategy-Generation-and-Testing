#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h Trend: EMA50 ===
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === 1d Volume filter: current volume > 20-period average ===
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # === 1d Previous day's Camarilla pivot points (HLC/3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_1d = prev_high_1d - prev_low_1d
    
    # Camarilla levels (R1, S1)
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe
    r1_1h = align_htf_to_ltf(prices, df_1d, r1)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1h Session filter: 08-20 UTC ===
    hours = prices.index.hour
    
    # === 1h Volume filter: current volume > 20-period average ===
    vol_ma20_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(vol_ma20_1d_aligned[i]) or np.isnan(vol_ma20_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price breaks above R1 with 4h uptrend and 1d volume confirmation
            long_cond = (close[i] > r1_1h[i] and 
                        close[i] > ema50_4h_aligned[i] and
                        volume[i] > vol_ma20_1h[i] and
                        vol_1d[i] > vol_ma20_1d_aligned[i] and
                        in_session)
            
            # Short: price breaks below S1 with 4h downtrend and 1d volume confirmation
            short_cond = (close[i] < s1_1h[i] and 
                         close[i] < ema50_4h_aligned[i] and
                         volume[i] > vol_ma20_1h[i] and
                         vol_1d[i] > vol_ma20_1d_aligned[i] and
                         in_session)
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or 4h trend turns down
            exit_cond = (close[i] < s1_1h[i] or 
                        close[i] < ema50_4h_aligned[i])
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R1 or 4h trend turns up
            exit_cond = (close[i] > r1_1h[i] or 
                        close[i] > ema50_4h_aligned[i])
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 1d volume confirmation.
# Enters long when price breaks above R1 in 4h uptrend with volume confirmation on both 1h and 1d.
# Enters short when price breaks below S1 in 4h downtrend with volume confirmation.
# Uses session filter (08-20 UTC) to avoid low-liquidity hours.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
# Uses discrete sizing (0.20) to reduce churn. Works in both bull (breakouts) and bear (trend continuation).