#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation (>2.0x 20-bar avg volume).
# Uses discrete sizing 0.28 to target 75-150 total trades over 4 years on 4h timeframe.
# Camarilla pivot levels provide strong intraday support/resistance; 1d EMA34 ensures higher timeframe trend alignment.
# Volume confirmation filters breakouts with low participation. Designed for fewer, higher-quality trades
# to minimize fee drag while working in both bull and bear markets.

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels (R3, S3) from prior 1d candle only
    lookback_camarilla = 1
    prior_close = pd.Series(close).rolling(window=lookback_camarilla, min_periods=lookback_camarilla).mean().shift(1).values
    prior_high = pd.Series(high).rolling(window=lookback_camarilla, min_periods=lookback_camarilla).max().shift(1).values
    prior_low = pd.Series(low).rolling(window=lookback_camarilla, min_periods=lookback_camarilla).min().shift(1).values
    
    # Camarilla R3 = C + (H-L) * 1.1/4
    # Camarilla S3 = C - (H-L) * 1.1/4
    camarilla_range = prior_high - prior_low
    camarilla_r3 = prior_close + camarilla_range * 1.1 / 4
    camarilla_s3 = prior_close - camarilla_range * 1.1 / 4
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_camarilla, lookback_vol, 1), n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(prior_close[i]) or np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1d EMA34, volume spike
            if (high[i] > camarilla_r3[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.28
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1d EMA34, volume spike
            elif (low[i] < camarilla_s3[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.28
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR volume drops below average
            if (low[i] < camarilla_s3[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR volume drops below average
            if (high[i] > camarilla_r3[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals