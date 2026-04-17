#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1-day ADX trend filter and volume confirmation.
- Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
- Long when Lips > Teeth > Jaw (bullish alignment) and price above 1-day ADX > 25 with volume > 1.5x 20-period volume MA
- Short when Lips < Teeth < Jaw (bearish alignment) and price below 1-day ADX > 25 with volume > 1.5x 20-period volume MA
- Exit when Alligator alignment reverses
- Fixed position size 0.25 to manage drawdown
- Uses 1-day ADX filter to avoid counter-trend trades in ranging markets
- Designed for 4h timeframe with strict entry conditions to limit trades to 75-200 total over 4 years
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (SMMA)"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=np.float64)
    result = np.full_like(series, np.nan, dtype=np.float64)
    # First value is simple SMA
    result[period-1] = np.mean(series[:period])
    # Subsequent values: (prev_smma * (period-1) + current_value) / period
    for i in range(period, len(series)):
        result[i] = (result[i-1] * (period-1) + series[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, +DM, -DM (14-period)
    tr_period = 14
    atr = np.full_like(tr, np.nan, dtype=np.float64)
    dm_plus_smooth = np.full_like(dm_plus, np.nan, dtype=np.float64)
    dm_minus_smooth = np.full_like(dm_minus, np.nan, dtype=np.float64)
    
    # Initial values (simple average)
    if len(tr) >= tr_period:
        atr[tr_period-1] = np.nanmean(tr[1:tr_period])  # Skip first NaN
        dm_plus_smooth[tr_period-1] = np.nanmean(dm_plus[1:tr_period])
        dm_minus_smooth[tr_period-1] = np.nanmean(dm_minus[1:tr_period])
        
        # Subsequent values (smoothed)
        for i in range(tr_period, len(tr)):
            atr[i] = (atr[i-1] * (tr_period-1) + tr[i]) / tr_period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period-1) + dm_plus[i]) / tr_period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period-1) + dm_minus[i]) / tr_period
    
    # Directional Indicators
    plus_di = np.full_like(atr, np.nan, dtype=np.float64)
    minus_di = np.full_like(atr, np.nan, dtype=np.float64)
    dx = np.full_like(atr, np.nan, dtype=np.float64)
    
    valid = (~np.isnan(atr)) & (atr != 0)
    plus_di[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
    minus_di[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
    
    dx_valid = (~np.isnan(plus_di)) & (~np.isnan(minus_di)) & ((plus_di + minus_di) != 0)
    dx[dx_valid] = 100 * np.abs(plus_di[dx_valid] - minus_di[dx_valid]) / (plus_di[dx_valid] + minus_di[dx_valid])
    
    # ADX (smoothed DX)
    adx = np.full_like(dx, np.nan, dtype=np.float64)
    adx_period = 14
    if len(dx) >= adx_period:
        # First ADX value is average of first adx_period DX values
        first_adx_idx = adx_period - 1
        if first_adx_idx < len(dx):
            adx[first_adx_idx] = np.nanmean(dx[1:adx_period+1])  # Skip first NaN in DX
            
            # Subsequent ADX values
            for i in range(adx_period+1, len(dx)):
                adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams Alligator components (using SMMA)
    # Jaw: 13-period SMMA shifted 8 bars
    jaw = smma(close, 13)
    jaw_shifted = np.roll(jaw, 8)
    jaw_shifted[:8] = np.nan  # First 8 values invalid after shift
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth = smma(close, 8)
    teeth_shifted = np.roll(teeth, 5)
    teeth_shifted[:5] = np.nan  # First 5 values invalid after shift
    
    # Lips: 5-period SMMA shifted 3 bars
    lips = smma(close, 5)
    lips_shifted = np.roll(lips, 3)
    lips_shifted[:3] = np.nan  # First 3 values invalid after shift
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # warmup for Alligator and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        lips_val = lips_shifted[i]
        teeth_val = teeth_shifted[i]
        jaw_val = jaw_shifted[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Look for Alligator signals with volume confirmation and trend filter
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_align = lips_val > teeth_val > jaw_val
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_align = lips_val < teeth_val < jaw_val
            
            # Long: Bullish alignment, ADX > 25 (trending), price above Jaw, volume spike
            if bullish_align and adx_val > 25 and price > jaw_val and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment, ADX > 25 (trending), price below Jaw, volume spike
            elif bearish_align and adx_val > 25 and price < jaw_val and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when Alligator turns bearish (Lips < Teeth or Teeth < Jaw)
            if lips_val < teeth_val or teeth_val < jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when Alligator turns bullish (Lips > Teeth or Teeth > Jaw)
            if lips_val > teeth_val or teeth_val > jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_ADX25_Volume"
timeframe = "4h"
leverage = 1.0