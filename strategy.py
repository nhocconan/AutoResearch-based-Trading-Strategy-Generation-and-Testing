#!/usr/bin/env python3
"""
1d_Williams_Alligator_Trend_With_Volume_Filter
Hypothesis: 1d Williams Alligator (SMMA(13,8), SMMA(8,5), SMMA(5,3)) defines trend (JAW > TEETH > LIPS = uptrend, JAW < TEETH < LIPS = downtrend). Enter on retests of TEETH line in trend direction with volume confirmation (>1.5x 20-bar MA). Exit on trend reversal (Alligator lines cross). Uses 1w EMA200 as higher timeframe filter to avoid counter-trend trades. Designed for low frequency (10-25 trades/year) to minimize fee drag while capturing sustained trends in both bull and bear markets via multi-timeframe alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) aka Wilder's MA"""
    if length < 1:
        return np.full_like(source, np.nan, dtype=np.float64)
    result = np.full_like(source, np.nan, dtype=np.float64)
    if len(source) < length:
        return result
    # First value is SMA
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + PRICE) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA200 filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # === 1d OHLC for Williams Alligator ===
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    df_1d_volume = df_1d['volume'].values
    
    # Calculate Williams Alligator lines (SMMA of median price)
    median_price = (df_1d_high + df_1d_low) / 2.0
    jaw = smma(median_price, 13)  # Blue line (13-period)
    teeth = smma(median_price, 8)  # Red line (8-period)
    lips = smma(median_price, 5)   # Green line (5-period)
    
    # Align 1d Alligator to 1d timeframe (no alignment needed as we're on 1d)
    # But we still use the helper for consistency and proper min_periods handling
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 1d Volume filter (1.5x 20-period MA) ===
    vol_ma = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1w EMA200 for higher timeframe trend filter ===
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):  # Start after warmup for Alligator and EMA200
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])
            or np.isnan(vol_ma[i]) or np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = df_1d_close[i]
        volume_now = df_1d_volume[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_avg = vol_ma[i]
        ema_200 = ema_200_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume_now > 1.5 * vol_avg
        
        # Alligator trend definition
        # Uptrend: JAW > TEETH > LIPS
        # Downtrend: JAW < TEETH < LIPS
        is_uptrend = jaw_val > teeth_val and teeth_val > lips_val
        is_downtrend = jaw_val < teeth_val and teeth_val < lips_val
        
        if position == 0:
            # Enter on retest of TEETH line in trend direction with volume confirmation
            # Long: price retests TEETH from below in uptrend
            long_condition = is_uptrend and volume_confirm and \
                           df_1d_low[i] <= teeth_val <= df_1d_high[i] and \
                           price > teeth_val and price > ema_200
            # Short: price retests TEETH from above in downtrend
            short_condition = is_downtrend and volume_confirm and \
                              df_1d_low[i] <= teeth_val <= df_1d_high[i] and \
                              price < teeth_val and price < ema_200
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit on trend reversal (Alligator lines cross) or price moves beyond JAW
            if position == 1:  # Long position
                # Exit if trend turns downtrend or price closes below JAW
                if not is_uptrend or price < jaw_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                # Exit if trend turns uptrend or price closes above JAW
                if not is_downtrend or price > jaw_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Williams_Alligator_Trend_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0