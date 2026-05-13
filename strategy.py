#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation (>1.8x 24-bar avg volume).
# Uses Camarilla pivot levels from 1d timeframe for structure, EMA50 for higher timeframe trend alignment,
# volume spike for participation confirmation. Designed for BTC/ETH with discrete sizing (0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years on 12h timeframe (12-37/year) to avoid fee drag.
# Strategy avoids overtrading by requiring strict confluence: breakout + trend + volume spike.

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_Volume_v1"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
    
    # Calculate average volume for confirmation (24-period on 12h = 12d lookback)
    lookback_vol = 24
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 1), n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1d EMA50, volume spike
            if (high[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25  # Position size 25% to manage drawdown
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1d EMA50, volume spike
            elif (low[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25  # Position size 25% to manage drawdown
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if breaks below Camarilla R3 or volume drops
            if low[i] < camarilla_r3_aligned[i] or volume[i] < 0.9 * avg_volume[i]:
                signals[i] = 0.0  # Exit to flat
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # EXIT SHORT: Close position if breaks above Camarilla S3 or volume drops
            if high[i] > camarilla_s3_aligned[i] or volume[i] < 0.9 * avg_volume[i]:
                signals[i] = 0.0  # Exit to flat
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals