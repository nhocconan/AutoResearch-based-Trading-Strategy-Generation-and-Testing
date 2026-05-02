#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide institutional reference points; breakout of R3/S3 indicates strong momentum
# 1d EMA34 ensures alignment with daily trend; volume spike (2.0x 20-period average) confirms participation
# Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years)
# Works in bull markets via R3 breakout continuation and in bear markets via S3 breakdown with trend filter

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4, etc.
    # Using standard Camarilla formula based on previous day's range
    df_1d['range'] = df_1d['high'] - df_1d['low']
    camarilla_r3 = df_1d['close'] + 1.1 * df_1d['range'] * 1.1 / 4
    camarilla_s3 = df_1d['close'] - 1.1 * df_1d['range'] * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla calculation and volume MA)
    start_idx = 20  # buffer for 20-period calculations
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 + 1d close > EMA34 + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + 1d close < EMA34 + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price drops below Camarilla S3 (reversal to support) or 1d trend breaks
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla R3 (reversal to resistance) or 1d trend breaks
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals