#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 with price > 1w EMA34 (bullish trend) and volume > 1.8x 20-bar average.
# Short when price breaks below Camarilla S3 with price < 1w EMA34 (bearish trend) and volume > 1.8x average.
# Exit when price reverses and closes below/above the Camarilla H3/L3 level (mean reversion exit).
# Uses discrete position sizing 0.25. Target: 50-150 total trades over 4 years on 12h timeframe.
# 1w EMA trend filter ensures we trade with the higher timeframe trend, avoiding counter-trend whipsaws.
# Volume confirmation validates breakout strength. Camarilla exit provides clear, objective stop.
# Camarilla pivot levels work well in both trending and ranging markets, providing clear breakout levels.

name = "12h_Camarilla_R3_S3_1wEMA34_Trend_VolumeConfirm"
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
    
    lookback = 20  # for volume average and pivot calculation
    
    # Calculate Camarilla pivot levels (based on previous bar's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    #          H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    prev_range = prev_high - prev_low
    
    camarilla_r3 = prev_close + 1.1 * prev_range
    camarilla_s3 = prev_close - 1.1 * prev_range
    camarilla_h3 = prev_close + 0.55 * prev_range  # 1.1/2 = 0.55
    camarilla_l3 = prev_close - 0.55 * prev_range  # 1.1/2 = 0.55
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1w close
    if len(close_1w) < 34:
        ema_34_1w = np.full(len(close_1w), np.nan)
    else:
        ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 12h timeframe (wait for 1w bar to close)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with bullish 1w EMA trend and volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 with bearish 1w EMA trend and volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla H3 (mean reversion)
            if close[i] < camarilla_h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla L3 (mean reversion)
            if close[i] > camarilla_l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals