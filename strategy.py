#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with 1d volume spike and 1w EMA50 trend filter
# Camarilla R4/S4 levels provide stronger breakout confirmation than R3/S3
# 1d volume spike (>2x 20-bar EMA) confirms institutional participation
# 1w EMA50 trend filter ensures alignment with weekly trend to avoid counter-trend whipsaws
# Designed for 12h timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Works in bull markets (breakout above R4 + 1w EMA50 up-trend) and bear markets (breakout below S4 + 1w EMA50 down-trend)

name = "12h_Camarilla_R4S4_Breakout_1dVolume_1wEMA50_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume EMA
        return np.zeros(n)
    
    # 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    # 1d volume EMA20 for confirmation
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)  # Volume > 2x EMA20
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels for each 1d bar (based on same day's OHLC)
    # Standard Camarilla: R4 = close + (high-low)*1.1/2, S4 = close - (high-low)*1.1/2
    camarilla_r4 = df_1d['close'].values + (df_1d['high'].values - df_1d['low'].values) * 1.1 / 2
    camarilla_s4 = df_1d['close'].values - (df_1d['high'].values - df_1d['low'].values) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (use same day's levels)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R4 with volume confirmation and uptrend
            if close[i] > camarilla_r4_aligned[i] and uptrend and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below S4 with volume confirmation and downtrend
            elif close[i] < camarilla_s4_aligned[i] and downtrend and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S4 (reversal) OR trend changes to downtrend
            if close[i] < camarilla_s4_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R4 (reversal) OR trend changes to uptrend
            if close[i] > camarilla_r4_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals