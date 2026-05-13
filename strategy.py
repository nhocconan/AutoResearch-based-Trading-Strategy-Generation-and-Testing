#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 and close > 4h EMA50 with volume > 1.8x 24-bar average.
# Short when price breaks below Camarilla S3 and close < 4h EMA50 with volume > 1.8x 24-bar average.
# Uses discrete sizing 0.20 to target 60-150 total trades over 4 years on 1h timeframe.
# Camarilla levels from prior 1h candle provide intraday structure; 4h EMA50 filters counter-trend noise; volume confirms momentum.
# Session filter (08-20 UTC) reduces noise trades. Designed for fewer, higher-quality trades to avoid fee drag while working in both bull and bear markets.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_Trend_VolumeConfirm"
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
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from prior 1h candle only
    lookback_1h = 1
    prior_high = pd.Series(high).rolling(window=lookback_1h, min_periods=lookback_1h).max().shift(1).values
    prior_low = pd.Series(low).rolling(window=lookback_1h, min_periods=lookback_1h).min().shift(1).values
    prior_close = pd.Series(close).rolling(window=lookback_1h, min_periods=lookback_1h).mean().shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_range = prior_high - prior_low
    camarilla_r3 = prior_close + camarilla_range * 1.1 / 4
    camarilla_s3 = prior_close - camarilla_range * 1.1 / 4
    
    # Calculate average volume for confirmation (24-period)
    lookback_vol = 24
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 1), n):  # Start after sufficient data
        # Skip if any required data is NaN or outside session
        if (np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or np.isnan(prior_close[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(avg_volume[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 4h EMA50, volume spike
            if (high[i] > camarilla_r3[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 4h EMA50, volume spike
            elif (low[i] < camarilla_s3[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.20
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
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR volume drops below average
            if (high[i] > camarilla_r3[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals