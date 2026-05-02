#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Camarilla levels provide institutional support/resistance from 1d timeframe
# 1d EMA34 determines trend bias: long when price > EMA34, short when price < EMA34
# Volume spike (2x 20-period average) confirms breakout validity
# Works in bull markets via breakouts with trend alignment and bear markets via fade of false breakouts
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (prior completed 1d bar's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior completed 1d bar's high, low, close for Camarilla
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    camarilla_h5 = prev_close + range_ * 1.1 / 2  # R3
    camarilla_h4 = prev_close + range_ * 1.1 / 4  # R2
    camarilla_h3 = prev_close + range_ * 1.1 / 6  # R1
    camarilla_l3 = prev_close - range_ * 1.1 / 6  # S1
    camarilla_l2 = prev_close - range_ * 1.1 / 4  # S2
    camarilla_l1 = prev_close - range_ * 1.1 / 2  # S3
    
    # Align to 6h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Calculate 1d EMA34 trend (prior completed 1d bar's EMA)
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 6h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND price > 1d EMA34 (bullish bias) AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 AND price < 1d EMA34 (bearish bias) AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below S3 OR below 1d EMA34 (trend change)
            if close[i] < s3_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above R3 OR above 1d EMA34 (trend change)
            if close[i] > r3_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals