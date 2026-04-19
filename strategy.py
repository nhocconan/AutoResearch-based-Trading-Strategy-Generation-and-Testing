#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Alligator + 1d price action with volume confirmation
# Uses Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) to filter trend direction
# Long when price > Lips and Teeth > Jaw (bullish alignment) with volume confirmation
# Short when price < Lips and Teeth < Jaw (bearish alignment) with volume confirmation
# Weekly trend filter ensures alignment with higher timeframe momentum
# Target: 15-30 trades/year to minimize fee drag while capturing major moves
name = "1d_Alligator_Trend_Filter_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for multi-timeframe trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly close for trend filter (only use if price > weekly close = uptrend bias)
    close_1w = df_1w['close'].values
    weekly_close = close_1w  # Already the weekly close series
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    
    # Williams Alligator components (using SMAs as per original)
    # Jaw: 13-period SMMA (smoothed MA) of median price, shifted 8 bars forward
    # Teeth: 8-period SMMA of median price, shifted 5 bars forward  
    # Lips: 5-period SMMA of median price, shifted 3 bars forward
    median_price = (high + low) / 2
    
    # Calculate SMMA (Smoothed Moving Average) - equivalent to RMA/Wilder's MA
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: (prev*(period-1) + current) / period
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Jaw (13, 8)
    jaw_raw = smma(median_price, 13)
    jaw = np.roll(jaw_raw, 8)  # Shift forward 8 bars
    jaw[:13+8] = np.nan  # Not enough data
    
    # Teeth (8, 5)
    teeth_raw = smma(median_price, 8)
    teeth = np.roll(teeth_raw, 5)  # Shift forward 5 bars
    teeth[:8+5] = np.nan  # Not enough data
    
    # Lips (5, 3)
    lips_raw = smma(median_price, 5)
    lips = np.roll(lips_raw, 3)  # Shift forward 3 bars
    lips[:5+3] = np.nan  # Not enough data
    
    # Align Alligator components to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # ATR for volatility filtering and position sizing
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or \
           np.isnan(weekly_close_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr_val = atr[i]
        
        # Volume filter: current volume > 1.3x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[max(0, i-20):i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.3 * avg_volume
        
        # Alligator alignment conditions
        # Bullish: price > Lips AND Teeth > Jaw
        # Bearish: price < Lips AND Teeth < Jaw
        bullish_alignment = (price > lips_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (price < lips_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Weekly trend filter: only take longs in weekly uptrend, shorts in weekly downtrend
        weekly_uptrend = price > weekly_close_aligned[i]
        weekly_downtrend = price < weekly_close_aligned[i]
        
        if position == 0:
            # Long: bullish alignment + volume + weekly uptrend
            if bullish_alignment and volume_filter and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + volume + weekly downtrend
            elif bearish_alignment and volume_filter and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below Lips or ATR-based trailing stop
            if price < lips_aligned[i] or price < np.maximum.accumulate(close[:i+1])[-1] - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above Lips or ATR-based trailing stop
            if price > lips_aligned[i] or price > np.minimum.accumulate(close[:i+1])[-1] + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals