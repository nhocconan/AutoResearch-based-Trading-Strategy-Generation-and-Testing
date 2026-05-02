#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter and volume spike confirmation
# Camarilla pivot levels (R3/S3) provide high-probability reversal/breakout zones derived from prior day's range
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades
# Volume spike (>2.0 x 20 EMA) confirms institutional participation in breakouts
# Works in bull markets (breakout above R3 + 1d EMA34 up) and bear markets (breakout below S3 + 1d EMA34 down)
# Uses discrete position sizing (0.25) to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
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
    
    # 1d data for trend filter (EMA34) and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 calculation for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from 1d timeframe (based on prior day's OHLC)
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using prior day's values to avoid look-ahead
    prior_close = df_1d['close'].shift(1).values
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    camarilla_r3 = prior_close + 1.1 * (prior_high - prior_low) / 2
    camarilla_s3 = prior_close - 1.1 * (prior_high - prior_low) / 2
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA and Camarilla calculation)
    start_idx = 35
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Camarilla R3 with volume confirmation and uptrend
            if high[i] > camarilla_r3_aligned[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below Camarilla S3 with volume confirmation and downtrend
            elif low[i] < camarilla_s3_aligned[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla S3 (reversal) OR trend changes to downtrend
            if low[i] < camarilla_s3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla R3 (reversal) OR trend changes to uptrend
            if high[i] > camarilla_r3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals