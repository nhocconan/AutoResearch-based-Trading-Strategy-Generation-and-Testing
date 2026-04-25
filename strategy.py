#!/usr/bin/env python3
"""
4h Camarilla R3S3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla R3/S3 levels act as intraday support/resistance derived from prior day's range.
Breaks above R3 or below S3 indicate strong momentum. Trend filter (price vs 1d EMA34) ensures
we trade with the higher timeframe direction. Volume spike confirms institutional participation.
Designed for fewer trades (target: 75-200 over 4 years) to minimize fee drag.
Works in bull markets (buy R3 breakouts in uptrend) and bear markets (sell S3 breakdowns in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar (HLC of previous daily bar)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # We use the prior completed daily bar to avoid look-ahead
    df_1d_shift = df_1d.shift(1)  # shift to get prior day's HLC
    prior_high = df_1d_shift['high'].values
    prior_low = df_1d_shift['low'].values
    prior_close = df_1d_shift['close'].values
    
    # Calculate Camarilla R3 and S3 from prior day
    rang = prior_high - prior_low
    camarilla_r3 = prior_close + 1.1 * rang * 1.1 / 4
    camarilla_s3 = prior_close - 1.1 * rang * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (they change only once per day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34)  # volume MA, 1d EMA34 alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_aligned[i]
        downtrend = curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Close breaks above R3 AND uptrend AND volume spike
            long_entry = (curr_close > curr_r3) and uptrend and vol_spike
            # Short: Close breaks below S3 AND downtrend AND volume spike
            short_entry = (curr_close < curr_s3) and downtrend and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Close drops below S3 (reversal) OR loss of uptrend
            if (curr_close < curr_s3) or (curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Close rises above R3 (reversal) OR loss of downtrend
            if (curr_close > curr_r3) or (curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0