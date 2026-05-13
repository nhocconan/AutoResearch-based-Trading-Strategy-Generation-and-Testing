#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (>1.5x 20-bar avg volume).
# Uses Camarilla pivot levels from 1d timeframe for structure, EMA34 for higher timeframe trend alignment, volume spike for participation confirmation.
# Designed for BTC/ETH with discrete sizing (0.30) to minimize fee churn while capturing strong momentum moves in both bull and bear markets.
# Target: 50-150 total trades over 4 years on 12h timeframe.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Volume_v1"
timeframe = "12h"
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (based on prior 1d bar)
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    prior_1d_high = df_1d['high'].values
    prior_1d_low = df_1d['low'].values
    prior_1d_close = df_1d['close'].values
    
    camarilla_r3 = prior_1d_close + 1.1 * (prior_1d_high - prior_1d_low) * 1.1 / 4
    camarilla_s3 = prior_1d_close - 1.1 * (prior_1d_high - prior_1d_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 1), n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1d EMA34, volume spike
            if (high[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.30  # Full position on breakout
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1d EMA34, volume spike
            elif (low[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.30  # Full position on breakout
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # CONTINUE LONG: Reduce to half position if still above R3 and volume OK
            if (high[i] > camarilla_r3_aligned[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = 0.15  # Half position to reduce churn
            else:
                signals[i] = 0.0  # Exit if breaks below R3 or low volume
                position = 0
        elif position == -1:
            # CONTINUE SHORT: Reduce to half position if still below S3 and volume OK
            if (low[i] < camarilla_s3_aligned[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = -0.15  # Half position to reduce churn
            else:
                signals[i] = 0.0  # Exit if breaks above S3 or low volume
                position = 0
    
    return signals