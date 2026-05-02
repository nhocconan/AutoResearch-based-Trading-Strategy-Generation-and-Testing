#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h trend filter and volume confirmation
# Camarilla pivot levels provide precise intraday support/resistance; breakouts above R3 or below S3 indicate strong momentum
# 12h EMA(34) ensures alignment with intermediate trend to avoid counter-trend trades
# Volume spike (1.8x 24-period average) confirms institutional participation
# Discrete position sizing (0.25) minimizes fee churn
# Targets 19-50 trades/year (75-200 total over 4 years) for 4h timeframe
# Works in bull markets via breakout continuation and in bear markets via filtered short breakdowns

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend filter
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 1d Camarilla pivot levels (using previous day's OHLC)
    # Camarilla levels: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #                 S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use R3/S3 for breakout signals
    lookback = 1  # previous day
    prev_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    prev_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    prev_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    # Calculate pivot levels
    pivot_range = prev_high - prev_low
    camarilla_r3 = prev_close + 1.1 * pivot_range
    camarilla_s3 = prev_close - 1.1 * pivot_range
    
    # Calculate volume spike (1.8x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for pivot calculation and volume MA)
    start_idx = 24 + 1  # buffer for 24-period volume MA and 1-day pivot
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 + 12h close > 12h EMA + volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_12h_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + 12h close < 12h EMA + volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_12h_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price drops below Camarilla S3 (reversal to mean) or 12h trend breaks
            if close[i] < camarilla_s3[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla R3 (reversal to mean) or 12h trend breaks
            if close[i] > camarilla_r3[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals