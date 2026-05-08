#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Reversal with 1d Trend Filter and Volume Spike
# - Uses daily Camarilla pivot levels (S1, S2, S3, R1, R2, R3) as key support/resistance
# - Long when price touches S1/S2 with bullish 1d trend and volume spike
# - Short when price touches R1/R2 with bearish 1d trend and volume spike
# - Works in bull/bear by using 1d trend filter to avoid counter-trend trades
# - Target: 15-30 trades/year on 12h to minimize fee drag
# - Timeframe: 12h, HTF: 1d for pivots and trend

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
    
    # Calculate Camarilla pivot levels for each day
    # Typical price = (high + low + close) / 3
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    # R4 = close + range * 1.500
    # R3 = close + range * 1.250
    # R2 = close + range * 1.166
    # R1 = close + range * 1.083
    # S1 = close - range * 1.083
    # S2 = close - range * 1.166
    # S3 = close - range * 1.250
    # S4 = close - range * 1.500
    r1_1d = close_1d + range_1d * 1.083
    r2_1d = close_1d + range_1d * 1.166
    s1_1d = close_1d - range_1d * 1.083
    s2_1d = close_1d - range_1d * 1.166
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0x 30-period average on 12h
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches S1 or S2 + 1d uptrend + volume spike
            long_cond = ((close[i] <= s1_1d_aligned[i] * 1.002 and close[i] >= s1_1d_aligned[i] * 0.998) or
                         (close[i] <= s2_1d_aligned[i] * 1.002 and close[i] >= s2_1d_aligned[i] * 0.998)) and \
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and \
                        volume_spike[i]
            
            # Short: price touches R1 or R2 + 1d downtrend + volume spike
            short_cond = ((close[i] <= r1_1d_aligned[i] * 1.002 and close[i] >= r1_1d_aligned[i] * 0.998) or
                          (close[i] <= r2_1d_aligned[i] * 1.002 and close[i] >= r2_1d_aligned[i] * 0.998)) and \
                         ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price touches R1 or R1 broken, or trend turns bearish
            if (close[i] >= r1_1d_aligned[i] * 0.998 or 
                ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches S1 or S1 broken, or trend turns bullish
            if (close[i] <= s1_1d_aligned[i] * 1.002 or 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals