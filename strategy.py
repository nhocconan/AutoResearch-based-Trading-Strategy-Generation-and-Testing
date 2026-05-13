#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 level with price > 1d EMA34 (bullish trend) and volume > 2.0x 24-bar average.
# Short when price breaks below Camarilla S3 level with price < 1d EMA34 (bearish trend) and volume > 2.0x average.
# Exit when price reverses and closes below/above the 1d VWAP (dynamic mean reversion exit).
# Uses discrete position sizing 0.28. Target: 80-180 total trades over 4 years on 4h timeframe.
# Camarilla pivot levels from 1d provide institutional support/resistance structure that works in both trending and ranging markets.
# EMA trend filter ensures we trade with the higher timeframe trend, avoiding counter-trend whipsaws.
# Volume confirmation validates breakout strength. VWAP exit provides adaptive stop that tightens in ranging markets.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
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
    
    lookback = 24  # for volume average and pivot calculation window
    
    # Get 1d data for EMA trend filter and VWAP exit
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(34) on 1d close
    if len(close_1d) < 34:
        ema_34_1d = np.full(len(close_1d), np.nan)
    else:
        ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d VWAP for exit (typical price * volume / cumulative volume)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    # Assume volume data is available in df_1d; if not, use close as proxy
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones_like(close_1d)
    vwap_1d = (typical_price_1d * volume_1d).cumsum() / volume_1d.cumsum()
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate Camarilla pivot levels (R3, S3) from previous 1d bar
    # R3 = close + 1.1*(high - low)/2, S3 = close - 1.1*(high - low)/2
    camarilla_r3_1d = close_1d + 1.1 * (high_1d - low_1d) / 2.0
    camarilla_s3_1d = close_1d - 1.1 * (high_1d - low_1d) / 2.0
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Calculate average volume for confirmation (24-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with bullish 1d EMA trend and volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.28
                position = 1
            # SHORT: Price breaks below Camarilla S3 with bearish 1d EMA trend and volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.28
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 1d VWAP (mean reversion)
            if close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # EXIT SHORT: Price closes above 1d VWAP (mean reversion)
            if close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals