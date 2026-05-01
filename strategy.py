#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla H3/L3 breakouts with 1d volume confirmation and session filter (08-20 UTC)
# Uses 4h for signal direction (Camarilla breakouts) and 1h only for entry timing precision
# Volume spike > 2.0x 20-period EMA reduces false breakouts
# Session filter (08-20 UTC) avoids low-liquidity periods
# Designed for optimal trade frequency: ~15-35 trades/year per symbol with 0.20 sizing
# Works in bull/bear: volume confirmation ensures participation, session filter reduces noise

name = "1h_Camarilla_H3L3_Breakout_1dVolume_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # 4h HTF data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 1d HTF data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h bar
    # H3 = close + 1.1*(high - low)/6
    # L3 = close - 1.1*(high - low)/6
    camarilla_H3 = df_4h['close'] + 1.1 * (df_4h['high'] - df_4h['low']) / 6
    camarilla_L3 = df_4h['close'] - 1.1 * (df_4h['high'] - df_4h['low']) / 6
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_H3.values)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_L3.values)
    
    # 1d volume spike filter: volume > 2.0 * 20-period EMA (tighter for fewer trades)
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Session filter: 08-20 UTC (avoid low-liquidity periods)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(20, 20)  # Need volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade during session hours
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Camarilla H3 with volume spike
            if close[i] > camarilla_H3_aligned[i] and volume_spike_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla L3 with volume spike
            elif close[i] < camarilla_L3_aligned[i] and volume_spike_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to Camarilla L3
            if close[i] <= camarilla_L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price returns to Camarilla H3
            if close[i] >= camarilla_H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals