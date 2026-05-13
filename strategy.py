#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter, volume confirmation (>1.5x 20-bar avg volume), and session filter (08-20 UTC). 
# Uses discrete sizing 0.20 to target 15-37 trades/year on 1h timeframe. 
# Camarilla levels provide high-probability intraday reversal/breakout zones; 
# 4h EMA50 ensures higher timeframe trend alignment to avoid counter-trend trades; 
# Volume confirmation filters low-participation breakouts; 
# Session filter avoids low-liquidity off-hours noise. 
# Designed for fewer, higher-quality trades to minimize fee drag while working in both bull and bear markets.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_Volume_Session"
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
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels (R3, S3) from prior day only
    lookback_camarilla = 24  # 24 * 1h = 1 day
    prior_close = pd.Series(close).shift(lookback_camarilla).values
    prior_high = pd.Series(high).rolling(window=lookback_camarilla, min_periods=lookback_camarilla).max().shift(lookback_camarilla).values
    prior_low = pd.Series(low).rolling(window=lookback_camarilla, min_periods=lookback_camarilla).min().shift(lookback_camarilla).values
    
    camarilla_range = prior_high - prior_low
    R3 = prior_close + camarilla_range * 1.1 / 4
    S3 = prior_close - camarilla_range * 1.1 / 4
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback_camarilla + 1, lookback_vol + 1, 1)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(prior_close[i]) or np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or
            np.isnan(R3[i]) or np.isnan(S3[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3, close > 4h EMA50, volume spike
            if (high[i] > R3[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S3, close < 4h EMA50, volume spike
            elif (low[i] < S3[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 OR volume drops below average
            if (low[i] < S3[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 OR volume drops below average
            if (high[i] > R3[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals