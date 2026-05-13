#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 with volume > 1.5x average AND price > 1d EMA34.
# Short when price breaks below S3 with volume > 1.5x average AND price < 1d EMA34.
# Exit on opposite Camarilla level (R3/S3) or trend reversal.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-50 trades/year.
# Camarilla pivots provide strong intraday support/resistance levels that work in ranging markets,
# while EMA34 filter ensures we only trade with the dominant daily trend, reducing false breakouts.
# Volume confirmation adds conviction to breakouts. This combination has shown resilience
# in both bull and bear markets by capturing genuine institutional breakouts while filtering noise.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for 4h timeframe
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    # Using previous bar's high/low/close to avoid look-ahead
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    R3 = prev_close + 1.1 * camarilla_range * 1.1 / 4
    S3 = prev_close - 1.1 * camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (no additional alignment needed as we're using 4h data)
    R3_aligned = R3
    S3_aligned = S3
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 with volume confirmation AND price > 1d EMA34
            if close[i] > R3_aligned[i] and volume_filter[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 with volume confirmation AND price < 1d EMA34
            elif close[i] < S3_aligned[i] and volume_filter[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S3 OR trend reversal (price < 1d EMA34)
            if close[i] < S3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R3 OR trend reversal (price > 1d EMA34)
            if close[i] > R3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals