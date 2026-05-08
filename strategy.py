#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Reversal with 1d Trend and Volume Spike
# - Camarilla pivot levels identify key support/resistance from prior day
# - Price reversal from S3/R3 levels with 1d trend alignment and volume spike
# - Works in bull/bear by using 1d trend filter to avoid counter-trend trades
# - Target: 12-37 trades/year to minimize fee drag on 12h timeframe

name = "12h_Camarilla_Pivot_Reversal_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R2 = C + ((H-L)*1.1/6), R1 = C + ((H-L)*1.1/12)
    # S1 = C - ((H-L)*1.1/12), S2 = C - ((H-L)*1.1/6), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    n1d = len(high_1d)
    camarilla_R3 = np.full(n1d, np.nan)
    camarilla_S3 = np.full(n1d, np.nan)
    
    for i in range(1, n1d):  # Start from 1 to have previous day data
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d[i-1]
        range_hl = H - L
        
        camarilla_R3[i] = C + (range_hl * 1.1 / 4)
        camarilla_S3[i] = C - (range_hl * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price reverses up from S3 support + 1d uptrend + volume spike
            long_cond = (close[i] < camarilla_S3_aligned[i] * 1.005 and  # Near S3
                        close[i] > camarilla_S3_aligned[i] and          # Above S3
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price reverses down from R3 resistance + 1d downtrend + volume spike
            short_cond = (close[i] > camarilla_R3_aligned[i] * 0.995 and  # Near R3
                         close[i] < camarilla_R3_aligned[i] and          # Below R3
                         ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches R3 or breaks below S3
            if close[i] >= camarilla_R3_aligned[i] or close[i] < camarilla_S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches S3 or breaks above R3
            if close[i] <= camarilla_S3_aligned[i] or close[i] > camarilla_R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals