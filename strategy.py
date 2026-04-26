#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike_v1
Hypothesis: On 1h timeframe, use 4h Camarilla R1/S1 breakouts for entry timing, 
filtered by 4h EMA(50) trend and 1d volume spike (2.5x median). 
Only trade during 08-20 UTC session to avoid low-liquidity hours.
Position size fixed at 0.20 to limit drawdown. 
4h provides signal direction and structure, 1h gives precise entry timing.
Target: 15-35 trades/year per symbol to avoid fee drag while capturing Camarilla breakouts with volume confirmation.
Works in bull markets (breakouts with trend) and bear markets (failed breakouts reverse quickly).
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
    
    # Get 4h data for HTF trend and Camarilla levels (primary signal source)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 4h OHLC
    prev_close_4h = df_4h['close'].shift(1).values
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    
    camarilla_r1 = prev_close_4h + (1.0/6) * (prev_high_4h - prev_low_4h)
    camarilla_s1 = prev_close_4h - (1.0/6) * (prev_high_4h - prev_low_4h)
    
    # Align HTF indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # 1d volume median for spike filter (20-period)
    vol_median_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).median().values
    vol_median_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_1d, additional_delay_bars=1)  # wait for daily close
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Warmup: max of EMA(50) 4h (50), Camarilla (need 1 previous bar), volume median (20)
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            # Outside session: flatten if needed
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_median_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_1d_val = vol_median_1d_aligned[i]
        
        # Trend filter
        uptrend = close_val > ema_50_4h_val
        downtrend = close_val < ema_50_4h_val
        
        # Volume spike condition
        volume_spike = volume_val > 2.5 * vol_median_1d_val
        
        if position == 0:
            # Long: break above R1 with volume spike, and uptrend
            long_signal = (close_val > camarilla_r1_aligned[i]) and \
                          volume_spike and \
                          uptrend
            
            # Short: break below S1 with volume spike, and downtrend
            short_signal = (close_val < camarilla_s1_aligned[i]) and \
                           volume_spike and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit on close below Camarilla S1 (mean reversion) or trend change
            if close_val < camarilla_s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit on close above Camarilla R1 (mean reversion) or trend change
            if close_val > camarilla_r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0