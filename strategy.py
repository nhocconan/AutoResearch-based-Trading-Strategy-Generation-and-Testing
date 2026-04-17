#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot R1/S1 breakout with volume confirmation and 12h EMA trend filter.
Long when price breaks above Camarilla R1 AND volume > 1.5x average AND 12h EMA34 > 12h EMA89 (uptrend).
Short when price breaks below Camarilla S1 AND volume > 1.5x average AND 12h EMA34 < 12h EMA89 (downtrend).
Exit when price reverts to Camarilla midpoint (close) OR 12h EMA flattens (|EMA34-EMA89| < 0.1% of price).
Uses 4h for Camarilla calculation and 12h for EMA filter to reduce whipsaw in ranging markets.
Camarilla levels provide precise intraday support/resistance, volume filters breakout validity,
EMA trend filter avoids false signals in chop. Works in bull markets (captures uptrends via R1 breakouts)
and bear markets (captures downtrends via S1 breakdowns). Target: 75-200 total trades over 4 years.
"""

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
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar (based on previous 4h bar)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # We use the previous 4h bar's high/low/close to calculate levels for current bar
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = high_4h[0]  # first bar: use same bar
    prev_low_4h[0] = low_4h[0]
    prev_close_4h[0] = close_4h[0]
    
    camarilla_upper = prev_close_4h + 1.1 * (prev_high_4h - prev_low_4h) / 12
    camarilla_lower = prev_close_4h - 1.1 * (prev_high_4h - prev_low_4h) / 12
    camarilla_mid = prev_close_4h  # midpoint is previous close
    
    # Get 12h data for EMA filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMAs (34 and 89)
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_12h = close_12h_series.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 4h Camarilla to 4h timeframe (no alignment needed for same timeframe)
    camarilla_upper_aligned = camarilla_upper
    camarilla_lower_aligned = camarilla_lower
    camarilla_mid_aligned = camarilla_mid
    
    # Align 12h EMAs to 4h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    ema_89_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_89_12h)
    
    # Volume average (20-period) on 4h
    volume_4h = df_4h['volume'].values
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_upper_aligned[i]) or np.isnan(camarilla_lower_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(ema_89_12h_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        cu = camarilla_upper_aligned[i]
        cl = camarilla_lower_aligned[i]
        cm = camarilla_mid_aligned[i]
        ema34 = ema_34_12h_aligned[i]
        ema89 = ema_89_12h_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        # Calculate EMA separation as percentage of price
        ema_sep_pct = abs(ema34 - ema89) / price if price != 0 else 0
        
        if position == 0:
            # Long: price > Camarilla R1 AND volume > 1.5x avg AND EMA34 > EMA89 (uptrend)
            if price > cu and vol > 1.5 * vol_ma and ema34 > ema89:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S1 AND volume > 1.5x avg AND EMA34 < EMA89 (downtrend)
            elif price < cl and vol > 1.5 * vol_ma and ema34 < ema89:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla midpoint OR EMA flattens (low trend strength)
            if price < cm or ema_sep_pct < 0.001:  # 0.1% threshold
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla midpoint OR EMA flattens (low trend strength)
            if price > cm or ema_sep_pct < 0.001:  # 0.1% threshold
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_12hEMA_Filter"
timeframe = "4h"
leverage = 1.0