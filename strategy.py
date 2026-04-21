#!/usr/bin/env python3
"""
1d_WilliamsAlligator_RegimeFilter_v1
Hypothesis: Williams Alligator (jaw/teeth/lips) on 1d timeframe indicates trend direction. 
Filter by 1w EMA34 trend and volume spike to avoid false signals. 
Enter long when lips > teeth > jaw (bullish alignment) and price above 1w EMA34 with volume confirmation.
Enter short when lips < teeth < jaw (bearish alignment) and price below 1w EMA34 with volume confirmation.
Use choppiness filter (CHOP > 61.8) to avoid ranging markets. 
ATR-based stoploss (2.5x) and discrete sizing (0.25). 
Designed to capture strong trends while avoiding whipsaws in sideways markets. 
Target: 30-80 total trades over 4 years (7-20/year) to minimize fee drag.
Works in both bull and bear markets via 1w trend alignment and strict entry filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1d OHLC for Williams Alligator calculation (based on previous 1d bar) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d timeframe
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars  
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Smoothed Moving Average (SMMA) approximation using EMA with alpha=1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_1d = smma(median_price_1d, 13)
    teeth_1d = smma(median_price_1d, 8)
    lips_1d = smma(median_price_1d, 5)
    
    # Shift as per Alligator definition
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    
    # Align 1d Alligator lines to 1d timeframe (no shift needed for primary timeframe)
    jaw_1d_aligned = jaw_1d
    teeth_1d_aligned = teeth_1d
    lips_1d_aligned = lips_1d
    
    # === 1w EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Choppiness Index filter (14-period) ===
    # CHOP > 61.8 = ranging market (avoid), CHOP < 38.2 = trending
    def choppiness_index(high, low, close, period=14):
        if len(high) < period:
            return np.full_like(high, np.nan)
        atr_period = []
        for i in range(len(high)):
            if i < 1:
                atr_period.append(high[i] - low[i])
            else:
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                atr_period.append(tr)
        
        atr_sum = pd.Series(atr_period).rolling(window=period, min_periods=period).sum().values
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        
        # Avoid division by zero
        range_hl = highest_high - lowest_low
        chop = np.where(
            (range_hl > 0) & ~np.isnan(atr_sum) & ~np.isnan(range_hl),
            100 * np.log10(atr_sum / np.log(period) / range_hl),
            100.0  # Default to high chop (ranging) when invalid
        )
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Warmup period for indicators
        # Skip if indicators not ready
        if (np.isnan(lips_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or np.isnan(jaw_1d_aligned[i]) 
            or np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        lips = lips_1d_aligned[i]
        teeth = teeth_1d_aligned[i]
        jaw = jaw_1d_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_avg = vol_ma[i]
        chop_value = chop[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        # Alligator alignment conditions
        bullish_alignment = (lips > teeth) and (teeth > jaw)
        bearish_alignment = (lips < teeth) and (teeth < jaw)
        
        # Choppiness filter: only trade when NOT ranging (CHOP < 61.8)
        not_ranging = chop_value < 61.8
        
        if position == 0:
            # Enter long: bullish alignment + price above 1w EMA + volume + not ranging
            long_condition = bullish_alignment and (price > ema_trend) and volume_confirmed and not_ranging
            # Enter short: bearish alignment + price below 1w EMA + volume + not ranging
            short_condition = bearish_alignment and (price < ema_trend) and volume_confirmed and not_ranging
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.5x ATR)
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit: Alligator loses bullish alignment
            elif not bullish_alignment:
                signals[i] = 0.0
                position = 0
            # Optional: exit if chop rises too high (market becoming ranging)
            elif chop_value > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.5x ATR)
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit: Alligator loses bearish alignment
            elif not bearish_alignment:
                signals[i] = 0.0
                position = 0
            # Optional: exit if chop rises too high (market becoming ranging)
            elif chop_value > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_RegimeFilter_v1"
timeframe = "1d"
leverage = 1.0