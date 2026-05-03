#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA(34) trend + volume spike
# Williams Alligator uses smoothed medians (Jaw, Teeth, Lips) to identify trends
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34 AND volume spike
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34 AND volume spike
# Exit when Alligator alignment breaks or price crosses 1d EMA34
# Designed for low trade frequency (12-37/year on 6h) with strong trend filtration
# Works in bull (Alligator alignment up + rising EMA) and bear (Alligator alignment down + falling EMA)

name = "6h_WilliamsAlligator_1dEMA34_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 6h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 6h: SMMA (Smoothed Moving Average) of median price
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Jaw: SMMA of median, period 13, shift 8
    # Teeth: SMMA of median, period 8, shift 5
    # Lips: SMMA of median, period 5, shift 3
    def smma(values, period, shift):
        """Smoothed Moving Average (similar to Wilder's smoothing)"""
        if len(values) < period:
            return np.full_like(values, np.nan)
        # First value is simple average
        result = np.full_like(values, np.nan)
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current value) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        # Apply shift
        if shift > 0:
            result = np.roll(result, shift)
            result[:shift] = np.nan
        return result
    
    jaw = smma(median_price, 13, 8)
    teeth = smma(median_price, 8, 5)
    lips = smma(median_price, 5, 3)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(13, 20) + 8  # Alligator jaw warmup + shift + volume MA
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_alignment = lips[i] > teeth[i] > jaw[i]
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_alignment = lips[i] < teeth[i] < jaw[i]
            
            # Long entry: bullish alignment, price > 1d EMA34, volume spike
            if (bullish_alignment and 
                close[i] > ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment, price < 1d EMA34, volume spike
            elif (bearish_alignment and 
                  close[i] < ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator alignment breaks bearish OR price < 1d EMA34
            bearish_alignment = lips[i] < teeth[i] < jaw[i]
            if bearish_alignment or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks bullish OR price > 1d EMA34
            bullish_alignment = lips[i] > teeth[i] > jaw[i]
            if bullish_alignment or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals