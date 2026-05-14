#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R4/S4 breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R4 (1d) AND close > 12h EMA50 AND volume > 2.5x average
# Short when price breaks below Camarilla S4 (1d) AND close < 12h EMA50 AND volume > 2.5x average
# Exit when price crosses Camarilla H4/L4 (mean reversion) OR trend reversal (price crosses 12h EMA50)
# Uses 4h timeframe with 12h trend filter for better noise reduction vs 1d, targeting 75-200 trades over 4 years.
# Camarilla R4/S4 levels are more extreme than R3/S3, reducing false breakouts; 12h EMA50 filters intermediate trend; volume spike confirms authenticity.

name = "4h_Camarilla_R4S4_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R4, S4, H4, L4) on 1d data (using previous bar's OHLC to avoid look-ahead)
    if len(high_1d) >= 1:
        # Use previous bar's OHLC to calculate today's Camarilla levels
        prev_high = pd.Series(high_1d).shift(1).values
        prev_low = pd.Series(low_1d).shift(1).values
        prev_close = pd.Series(close_1d).shift(1).values
        
        # Calculate pivot point
        pivot = (prev_high + prev_low + prev_close) / 3
        range_hl = prev_high - prev_low
        
        # Camarilla levels (R4/S4 are more extreme, H4/L4 for mean reversion exit)
        r4 = pivot + range_hl * 1.1 / 2
        s4 = pivot - range_hl * 1.1 / 2
        h4 = pivot + range_hl * 1.1 / 4
        l4 = pivot - range_hl * 1.1 / 4
    else:
        r4 = np.full_like(high_1d, np.nan)
        s4 = np.full_like(low_1d, np.nan)
        h4 = np.full_like(high_1d, np.nan)
        l4 = np.full_like(low_1d, np.nan)
    
    # Align Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Get 12h data for EMA50 trend filter (MTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h data
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current 4h volume > 2.5x 20-period average (spike confirmation)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.5 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after sufficient data for EMA and volume
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > Camarilla R4 AND close > 12h EMA50 AND volume spike
            if close[i] > r4_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < Camarilla S4 AND close < 12h EMA50 AND volume spike
            elif close[i] < s4_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < Camarilla H4 (mean reversion) OR trend reversal (close < 12h EMA50)
            if close[i] < h4_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > Camarilla L4 (mean reversion) OR trend reversal (close > 12h EMA50)
            if close[i] > l4_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals