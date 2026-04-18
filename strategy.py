#!/usr/bin/env python3
"""
4h_Williams_Alligator_Trend_v1
Hypothesis: Use Williams Alligator (SMMA-based) on 4h for trend direction, confirmed by price position relative to SMMA lines. 
Go long when price > Alligator Jaw (13-period SMMA shifted 8 bars) and Jaw > Teeth (8-period SMMA shifted 5 bars) and Teeth > Lips (5-period SMMA shifted 3 bars).
Short when price < Lips and Lips < Teeth and Teeth < Jaw. 
Requires volume > 1.5x 20-period average for confirmation and 1d ADX > 25 to ensure trending market.
Exit when Alligator lines re-cross (trend weakness) or volume drops below average.
Target: 20-40 trades/year by requiring strong alignment and volume confirmation. Works in bull via long trends and bear via short trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - same as Wilder's smoothing"""
    if length < 1:
        return np.full_like(source, np.nan)
    result = np.full_like(source, np.nan)
    if len(source) >= length:
        # First value is simple average
        result[length-1] = np.mean(source[:length])
        # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
        for i in range(length, len(source)):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams Alligator
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Williams Alligator components on 4h
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars  
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price_4h = (high_4h + low_4h) / 2 if 'high_4h' in df_4h and 'low_4h' in df_4h else close_4h
    jaw_raw = smma(median_price_4h, 13)
    teeth_raw = smma(median_price_4h, 8)
    lips_raw = smma(median_price_4h, 5)
    
    # Apply shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw_shifted = np.full_like(jaw_raw, np.nan)
    teeth_shifted = np.full_like(teeth_raw, np.nan)
    lips_shifted = np.full_like(lips_raw, np.nan)
    
    if len(jaw_raw) > 8:
        jaw_shifted[8:] = jaw_raw[:-8]
    if len(teeth_raw) > 5:
        teeth_shifted[5:] = teeth_raw[:-5]
    if len(lips_raw) > 3:
        lips_shifted[3:] = lips_raw[:-3]
    
    # Align Alligator lines to 4h timeframe (no additional delay needed for SMMA)
    jaw_4h_aligned = align_htf_to_ltf(prices, df_4h, jaw_shifted)
    teeth_4h_aligned = align_htf_to_ltf(prices, df_4h, teeth_shifted)
    lips_4h_aligned = align_htf_to_ltf(prices, df_4h, lips_shifted)
    
    # Get 1d data for ADX (trend strength filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original index
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        if len(tr) >= period:
            # Initial values
            atr[period] = np.nanmean(tr[1:period+1])
            dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
            dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
            
            # Wilder smoothing
            for i in range(period + 1, len(tr)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
        
        # DI and DX
        di_plus = np.full_like(dm_plus_smooth, np.nan)
        di_minus = np.full_like(dm_minus_smooth, np.nan)
        dx = np.full_like(tr, np.nan)
        
        mask = atr != 0
        di_plus[mask] = 100 * dm_plus_smooth[mask] / atr[mask]
        di_minus[mask] = 100 * dm_minus_smooth[mask] / atr[mask]
        
        dx_mask = (di_plus + di_minus) != 0
        dx[dx_mask] = 100 * np.abs(di_plus[dx_mask] - di_minus[dx_mask]) / (di_plus[dx_mask] + di_minus[dx_mask])
        
        # ADX: smoothed DX
        adx = np.full_like(dx, np.nan)
        if len(dx) >= period:
            valid_dx = dx[~np.isnan(dx)]
            if len(valid_dx) >= period:
                # First ADX value is average of first 'period' DX values
                adx[period-1] = np.nanmean(dx[np.isnan(dx) == False][:period]) if np.sum(np.isnan(dx) == False) >= period else np.nan
                # Wilder smoothing for subsequent values
                for i in range(period, len(dx)):
                    if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
                        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
                    else:
                        adx[i] = np.nan
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14) + 1  # volume period and ADX period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_4h_aligned[i]) or np.isnan(teeth_4h_aligned[i]) or 
            np.isnan(lips_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend alignment checks
        jaw_above_teeth = jaw_4h_aligned[i] > teeth_4h_aligned[i]
        teeth_above_lips = teeth_4h_aligned[i] > lips_4h_aligned[i]
        lips_above_jaw = lips_4h_aligned[i] > jaw_4h_aligned[i]  # for short condition
        
        # Price position relative to Alligator
        price_above_jaw = close[i] > jaw_4h_aligned[i]
        price_below_lips = close[i] < lips_4h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # ADX trend strength filter
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 0 and strong_trend:
            # Long: price above Jaw AND Jaw > Teeth > Lips (perfect alignment)
            if price_above_jaw and jaw_above_teeth and teeth_above_lips and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price below Lips AND Lips < Teeth < Jaw (perfect alignment)
            elif price_below_lips and lips_above_jaw and teeth_above_lips and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend weakening (Jaw <= Teeth or Teeth <= Lips) or price crosses below Teeth
            if not (jaw_above_teeth and teeth_above_lips) or close[i] < teeth_4h_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend weakening (Lips <= Jaw or Teeth <= Lips) or price crosses above Teeth
            if not (lips_above_jaw and teeth_above_lips) or close[i] > teeth_4h_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_Trend_v1"
timeframe = "4h"
leverage = 1.0