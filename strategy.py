#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R4 and close > 1d EMA50 with volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S4 and close < 1d EMA50 with volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 4h timeframe.
# Camarilla levels from 1d provide intraday structure; 1d EMA50 filters counter-trend noise; volume confirms momentum.
# Designed for fewer, higher-quality trades to avoid fee drag while working in both bull and bear markets.

name = "4h_Camarilla_R4_S4_Breakout_1dEMA50_Trend_VolumeConfirm"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from prior 1d candle only
    lookback_1d = 1
    prior_high = pd.Series(high).rolling(window=lookback_1d, min_periods=lookback_1d).max().shift(1).values
    prior_low = pd.Series(low).rolling(window=lookback_1d, min_periods=lookback_1d).min().shift(1).values
    prior_close = pd.Series(close).rolling(window=lookback_1d, min_periods=lookback_1d).mean().shift(1).values
    
    # Camarilla R4 and S4 levels (using 1.1/2 multiplier for R4/S4)
    camarilla_range = prior_high - prior_low
    camarilla_r4 = prior_close + camarilla_range * 1.1 / 2
    camarilla_s4 = prior_close - camarilla_range * 1.1 / 2
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 1), n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or np.isnan(prior_close[i]) or
            np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R4, close > 1d EMA50, volume spike
            if (high[i] > camarilla_r4[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S4, close < 1d EMA50, volume spike
            elif (low[i] < camarilla_s4[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S4 OR volume drops below average
            if (low[i] < camarilla_s4[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R4 OR volume drops below average
            if (high[i] > camarilla_r4[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals