#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation (>1.8x 20-bar avg volume).
# Uses Camarilla pivot levels from 4h timeframe for structure, EMA50 for higher timeframe trend alignment, volume spike for participation confirmation.
# Session filter (08-20 UTC) to reduce noise. Discrete sizing (0.20) to minimize fee churn.
# Target: 60-150 total trades over 4 years on 1h timeframe (15-37/year).

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla levels (based on prior 4h bar)
    # R3 = close + 1.1*(high-low)*1.125/4
    # S3 = close - 1.1*(high-low)*1.125/4
    prior_4h_high = df_4h['high'].values
    prior_4h_low = df_4h['low'].values
    prior_4h_close = df_4h['close'].values
    
    camarilla_r3 = prior_4h_close + 1.1 * (prior_4h_high - prior_4h_low) * 1.125 / 4
    camarilla_s3 = prior_4h_close - 1.1 * (prior_4h_high - prior_4h_low) * 1.125 / 4
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 1), n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(avg_volume[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 4h EMA50, volume spike
            if (high[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.20  # Full position on breakout
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 4h EMA50, volume spike
            elif (low[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.20  # Full position on breakout
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # CONTINUE LONG: Reduce to half position if still above R3 and volume OK
            if (high[i] > camarilla_r3_aligned[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = 0.10  # Half position to reduce churn
            else:
                signals[i] = 0.0  # Exit if breaks below R3 or low volume
                position = 0
        elif position == -1:
            # CONTINUE SHORT: Reduce to half position if still below S3 and volume OK
            if (low[i] < camarilla_s3_aligned[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = -0.10  # Half position to reduce churn
            else:
                signals[i] = 0.0  # Exit if breaks above S3 or low volume
                position = 0
    
    return signals