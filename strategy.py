#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout + volume confirmation + 12h EMA34 trend filter.
Long when price breaks above Camarilla R1 AND volume > 1.3x average AND 12h EMA34 > prior EMA34.
Short when price breaks below Camarilla S1 AND volume > 1.3x average AND 12h EMA34 < prior EMA34.
Exit when price reverts to Camarilla H5/L5 or volume < 0.8x average.
Uses 4h for Camarilla calculation and 12h for EMA trend filter to reduce whipsaw.
Target: 75-200 total trades over 4 years (19-50/year). Camarilla levels provide precise intraday
support/resistance, volume confirms institutional participation, EMA filter ensures trend alignment.
Works in bull markets (captures uptrends via R1 breakouts) and bear markets (captures downtrends via S1 breaks).
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
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate previous day's OHLC for Camarilla levels
    # Camarilla uses prior day's range to calculate support/resistance
    prev_close = np.roll(close_4h, 1)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close[0] = close_4h[0]  # first period
    prev_high[0] = high_4h[0]
    prev_low[0] = low_4h[0]
    
    # Camarilla levels based on prior day
    range_ = prev_high - prev_low
    camarilla_h5 = prev_close + range_ * 1.1 / 2
    camarilla_h4 = prev_close + range_ * 1.1 / 4
    camarilla_h3 = prev_close + range_ * 1.1 / 6
    camarilla_h2 = prev_close + range_ * 1.1 / 12
    camarilla_h1 = prev_close + range_ * 1.1 / 24
    camarilla_l1 = prev_close - range_ * 1.1 / 24
    camarilla_l2 = prev_close - range_ * 1.1 / 12
    camarilla_l3 = prev_close - range_ * 1.1 / 6
    camarilla_l4 = prev_close - range_ * 1.1 / 4
    camarilla_l5 = prev_close - range_ * 1.1 / 2
    
    # Key levels: R1 = H1, S1 = L1, H5 = stop/reversal, L5 = stop/reversal
    r1 = camarilla_h1
    s1 = camarilla_l1
    h5 = camarilla_h5
    l5 = camarilla_l5
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume average (20-period) on 4h
    volume_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    # Align 4h Camarilla levels to 4h timeframe (no alignment needed)
    r1_aligned = r1
    s1_aligned = s1
    h5_aligned = h5
    l5_aligned = l5
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(h5_aligned[i]) or np.isnan(l5_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        h5_val = h5_aligned[i]
        l5_val = l5_aligned[i]
        ema_12h = ema_34_12h_aligned[i]
        vol_ma = volume_ma_4h_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > R1 AND volume > 1.3x avg AND 12h EMA34 rising
            if price > r1_val and vol > 1.3 * vol_ma and ema_12h > ema_34_12h_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price < S1 AND volume > 1.3x avg AND 12h EMA34 falling
            elif price < s1_val and vol > 1.3 * vol_ma and ema_12h < ema_34_12h_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < H5 OR volume < 0.8x avg (loss of momentum)
            if price < h5_val or vol < 0.8 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > L5 OR volume < 0.8x avg (loss of momentum)
            if price > l5_val or vol < 0.8 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_12hEMA34_Trend"
timeframe = "4h"
leverage = 1.0