#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 Breakout + 1w EMA50 Trend + Volume Spike Confirmation
# Camarilla levels identify key intraday support/resistance; breakout signals momentum
# Only trade breakouts in direction of 1w EMA50 trend to avoid false breakouts
# Volume spike (2.0x 20-period average) confirms institutional participation
# Discrete position sizing (0.25) minimizes fee churn
# Targets 7-25 trades/year (30-100 total over 4 years) for 1d timeframe
# Works in bull markets via buying breakouts in uptrend and in bear markets via selling breakdowns in downtrend

name = "1d_Camarilla_R3_S3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
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
    open_ = prices['open'].values  # needed for Camarilla calculation
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: based on previous day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_open = np.roll(open_, 1)
    
    # First bar has no previous data
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    prev_open[0] = open_[0]
    
    # Calculate Camarilla levels
    # R3 = Close + (High - Low) * 1.1/2
    # S3 = Close - (High - Low) * 1.1/2
    rang = prev_high - prev_low
    camarilla_r3 = prev_close + rang * 1.1 / 2
    camarilla_s3 = prev_close - rang * 1.1 / 2
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for volume MA and Camarilla)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: close breaks above R3 + 1w close > EMA50 (uptrend) + volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: close breaks below S3 + 1w close < EMA50 (downtrend) + volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: close breaks below S3 or 1w trend breaks
            if close[i] < camarilla_s3[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: close breaks above R3 or 1w trend breaks
            if close[i] > camarilla_r3[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals