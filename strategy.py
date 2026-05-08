#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d Trend Filter and Volume Spike
# - Williams Alligator (Jaw/Teeth/Lips) identifies trend direction
# - Trade only when price is outside Alligator's mouth + 1d trend alignment
# - Volume spike confirms breakout strength
# - Works in bull/bear by using 1d trend filter to avoid counter-trend trades
# - Target: 20-40 trades/year to minimize fee drag on 4h timeframe

name = "4h_Williams_Alligator_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Williams Alligator (SMA-based)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Williams Alligator lines (all SMAs)
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values  # Blue line (13-period)
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values   # Red line (8-period)
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values    # Green line (5-period)
    
    # Align Alligator lines to 4h timeframe (wait for 1d bar to close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator: lips > teeth > jaw = uptrend, lips < teeth < jaw = downtrend
        # Price outside mouth confirms trend strength
        if position == 0:
            # Long: price above lips AND lips > teeth > jaw (uptrend aligned) + 1d uptrend + volume spike
            long_cond = (close[i] > lips_aligned[i] and 
                        lips_aligned[i] > teeth_aligned[i] and 
                        teeth_aligned[i] > jaw_aligned[i] and
                        ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price below lips AND lips < teeth < jaw (downtrend aligned) + 1d downtrend + volume spike
            short_cond = (close[i] < lips_aligned[i] and 
                         lips_aligned[i] < teeth_aligned[i] and 
                         teeth_aligned[i] < jaw_aligned[i] and
                         ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below teeth (trend weakening)
            if close[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above teeth (trend weakening)
            if close[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals