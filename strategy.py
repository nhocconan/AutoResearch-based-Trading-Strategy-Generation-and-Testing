#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA(34) trend filter and volume spike confirmation.
- Primary timeframe: 12h for entries/exits.
- HTF: 1d EMA(34) for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3).
- Volume: Current 12h volume > 2.0 * 20-period volume MA to confirm breakout strength.
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND price > EMA34 AND volume spike.
         Short when Lips < Teeth < Jaw (bearish alignment) AND price < EMA34 AND volume spike.
- Exit: Opposite Alligator alignment or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Why it should work: Alligator identifies trends with smoothed averages reducing whipsaw;
                      EMA34 filters counter-trend noise; volume spike confirms institutional interest.
                      Works in bull (catch trends) and bear (avoid false signals in ranging markets).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=float)
    alpha = 1.0 / length
    # First value is simple average
    if len(source) >= length:
        result[length-1] = np.mean(source[:length])
        for i in range(length, len(source)):
            if not np.isnan(source[i]):
                result[i] = alpha * source[i] + (1 - alpha) * result[i-1]
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    
    # Alligator components: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw_raw = smma(median_12h, 13)
    teeth_raw = smma(median_12h, 8)
    lips_raw = smma(median_12h, 5)
    
    # Apply smoothing offsets: Jaw +8, Teeth +5, Lips +3
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Calculate 1d EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 12h for confirmation
    vol_ma_20 = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Williams Alligator signals: bullish when Lips > Teeth > Jaw
    bullish_align = (lips_aligned > teeth_aligned) & (teeth_aligned > jaw_aligned)
    # Bearish when Lips < Teeth < Jaw
    bearish_align = (lips_aligned < teeth_aligned) & (teeth_aligned < jaw_aligned)
    
    # Volume confirmation: current 12h volume > 2.0 * 20-period volume MA
    volume_spike = volume > (2.0 * vol_ma_20_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34)  # Need enough bars for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish entry: Lips > Teeth > Jaw AND price > EMA34 (bullish trend)
                if bullish_align[i] and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Lips < Teeth < Jaw AND price < EMA34 (bearish trend)
                elif bearish_align[i] and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Alligator alignment turns bearish OR loss of volume confirmation
            if not bullish_align[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment turns bullish OR loss of volume confirmation
            if not bearish_align[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0