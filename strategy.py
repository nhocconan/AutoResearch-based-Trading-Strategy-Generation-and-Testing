#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation + session filter (08-20 UTC)
# Uses Camarilla pivot levels (R3/S3) from 1d data for institutional breakout/breakdown signals
# 4h EMA50 ensures alignment with higher timeframe trend direction (works in bull/bear via filtered signals)
# Volume spike (2.0x 20-period average) confirms institutional participation
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Discrete position sizing (0.20) minimizes fee churn
# Targets 15-37 trades/year (60-150 total over 4 years) for 1h timeframe
# Works in bull markets via R3 breakout continuation and in bear markets via S3 breakdown continuation

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for Camarilla pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # Using 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (extra delay not needed as these are based on completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for volume MA and aligned indicators)
    start_idx = 20  # buffer for 20-period calculations
    
    for i in range(start_idx, n):
        # Skip if not in trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 + 4h close > EMA50 + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_4h_aligned[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 + 4h close < EMA50 + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_4h_aligned[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price drops below Camarilla S3 (reversal to support) or 4h trend breaks
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla R3 (reversal to resistance) or 4h trend breaks
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals