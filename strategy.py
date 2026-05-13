#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 with price > 1d EMA34 (bullish trend) and volume > 1.5x 20-bar average.
# Short when price breaks below Camarilla S3 with price < 1d EMA34 (bearish trend) and volume > 1.5x average.
# Exit when price reverses and closes back inside the Camarilla H3/L3 range (mean reversion exit).
# Uses discrete position sizing 0.25. Target: 50-150 total trades over 4 years on 6h timeframe.
# EMA trend filter ensures we trade with the higher timeframe trend, avoiding counter-trend whipsaws.
# Volume confirmation validates breakout strength. Camarilla exit provides clear, objective stop.
# Camarilla pivot levels work well in both trending and ranging markets, providing clear breakout/fade levels.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeConfirm"
timeframe = "6h"
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
    
    # Calculate Camarilla pivot levels from previous bar
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1/2
    # S3 = PP - (H - L) * 1.1/2
    # H3 = PP + (H - L) * 1.1/4
    # L3 = PP - (H - L) * 1.1/4
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    
    pp = (prev_high + prev_low + prev_close) / 3.0
    r3 = pp + (prev_high - prev_low) * 1.1 / 2.0
    s3 = pp - (prev_high - prev_low) * 1.1 / 2.0
    h3 = pp + (prev_high - prev_low) * 1.1 / 4.0
    l3 = pp - (prev_high - prev_low) * 1.1 / 4.0
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close
    if len(close_1d) < 34:
        ema_34_1d = np.full(len(close_1d), np.nan)
    else:
        ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 6h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with bullish 1d EMA trend and volume spike
            if (close[i] > r3[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 with bearish 1d EMA trend and volume spike
            elif (close[i] < s3[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes back inside H3/L3 range (mean reversion)
            if close[i] < h3[i] and close[i] > l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes back inside H3/L3 range (mean reversion)
            if close[i] < h3[i] and close[i] > l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals