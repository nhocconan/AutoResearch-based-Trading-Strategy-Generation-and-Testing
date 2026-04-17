#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND close > 4h EMA34 AND volume > 1.3x average.
Short when price breaks below Camarilla S1 AND close < 4h EMA34 AND volume > 1.3x average.
Exit when price reverts to Camarilla pivot (PP) OR volume drops below average.
Uses 1h for entry timing precision, 4h for trend direction to reduce whipsaw.
Target: 80-150 total trades over 4 years (20-38/year). Camarilla levels provide intraday
support/resistance, volume confirms breakout validity, EMA filter avoids counter-trend trades.
Works in bull markets (buys breakouts in uptrends) and bear markets (sells breakdowns in downtrends).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for Camarilla calculation
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate previous day's Camarilla levels on 1h timeframe
    # Use prior 24h period (96 bars for 1h) for OHLC
    lookback = 96  # 24 hours * 4 bars per hour = 96 bars for prior day
    if len(high_1h) < lookback + 20:  # need extra for Camarilla calculation
        return np.zeros(n)
    
    # Calculate Camarilla for each bar using prior day's OHLC
    camarilla_pp = np.full(n, np.nan)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(lookback, len(high_1h)):
        # Prior day's OHLC (24 hours ago)
        prior_high = np.max(high_1h[i-lookback:i])
        prior_low = np.min(low_1h[i-lookback:i])
        prior_close = close_1h[i-lookback]  # close 24 hours ago
        
        # Camarilla calculations
        rng = prior_high - prior_low
        camarilla_pp[i] = (prior_high + prior_low + prior_close) / 3
        camarilla_r1[i] = camarilla_pp[i] + (rng * 1.1 / 12)
        camarilla_s1[i] = camarilla_pp[i] - (rng * 1.1 / 12)
    
    # Get 4h data for EMA filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA34 on 4h timeframe
    close_4h_series = pd.Series(close_4h)
    ema_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h EMA to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume average (20-period) on 1h
    volume_1h = df_1h['volume'].values
    volume_ma = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = lookback + 20  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(camarilla_pp[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = camarilla_r1[i]
        s1 = camarilla_s1[i]
        pp = camarilla_pp[i]
        ema_val = ema_4h_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R1 AND close > 4h EMA34 AND volume > 1.3x avg
            if price > r1 and close[i] > ema_val and vol > 1.3 * vol_ma:
                signals[i] = 0.20
                position = 1
            # Short: price < Camarilla S1 AND close < 4h EMA34 AND volume > 1.3x avg
            elif price < s1 and close[i] < ema_val and vol > 1.3 * vol_ma:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla PP OR volume < average
            if price < pp or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price > Camarilla PP OR volume < average
            if price > pp or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Volume_EMA_Filter"
timeframe = "1h"
leverage = 1.0