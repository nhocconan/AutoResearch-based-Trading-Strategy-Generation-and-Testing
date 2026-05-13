#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation (>1.8x 20-bar avg volume).
# Uses Camarilla pivot levels from 12h timeframe for stronger structure, EMA50 for higher timeframe trend alignment, volume spike for participation confirmation.
# Designed for BTC/ETH with discrete sizing (0.25) to minimize fee churn while capturing strong momentum moves in both bull and bear markets.
# Target: 75-150 total trades over 4 years on 4h timeframe.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Volume_v1"
timeframe = "4h"
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h Camarilla levels (based on prior 12h bar)
    # R3 = close + 1.1*(high-low)*1.125/4
    # S3 = close - 1.1*(high-low)*1.125/4
    prior_12h_high = df_12h['high'].values
    prior_12h_low = df_12h['low'].values
    prior_12h_close = df_12h['close'].values
    
    camarilla_r3 = prior_12h_close + 1.1 * (prior_12h_high - prior_12h_low) * 1.125 / 4
    camarilla_s3 = prior_12h_close - 1.1 * (prior_12h_high - prior_12h_low) * 1.125 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 12h EMA50, volume spike
            if (high[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25  # Full position on breakout
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 12h EMA50, volume spike
            elif (low[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25  # Full position on breakout
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # CONTINUE LONG: Maintain full position if still above R3 and volume OK
            if (high[i] > camarilla_r3_aligned[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = 0.25  # Full position
            else:
                signals[i] = 0.0  # Exit if breaks below R3 or low volume
                position = 0
        elif position == -1:
            # CONTINUE SHORT: Maintain full position if still below S3 and volume OK
            if (low[i] < camarilla_s3_aligned[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = -0.25  # Full position
            else:
                signals[i] = 0.0  # Exit if breaks above S3 or low volume
                position = 0
    
    return signals