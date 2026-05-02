#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Camarilla R3/S3 levels represent strong intraday support/resistance where institutional order flow often accumulates
# Breaks above R3 or below S3 with volume indicate significant momentum shifts
# 1w EMA34 provides higher timeframe trend filter to reduce counter-trend whipsaws in both bull and bear markets
# Volume spike (2.0x 20-period average) confirms breakout conviction
# Targets 30-100 trades over 4 years (7-25/year) for 1d timeframe
# Works in bull markets by capturing breakouts with trend, works in bear markets by fading false breaks against trend

name = "1d_Camarilla_R3_S3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla: R3 = close + 1.500*(high-low), S3 = close - 1.500*(high-low)
    # We use the previous completed 1d bar to avoid look-ahead
    high_1d = df_1w['high'].values  # Using 1w high/low as proxy for 1d calculation (will be aligned)
    low_1d = df_1w['low'].values
    close_1d = df_1w['close'].values
    
    camarilla_high = close_1d + 1.500 * (high_1d - low_1d)  # R3 level
    camarilla_low = close_1d - 1.500 * (high_1d - low_1d)   # S3 level
    
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1w, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1w, camarilla_low)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for volume MA and HTF data alignment)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla R3 + price > 1w EMA34 + volume spike
            if close[i] > camarilla_high_aligned[i] and close[i] > ema_34_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 + price < 1w EMA34 + volume spike
            elif close[i] < camarilla_low_aligned[i] and close[i] < ema_34_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla S3 (reversal signal)
            if close[i] < camarilla_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla R3 (reversal signal)
            if close[i] > camarilla_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals