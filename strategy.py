#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with weekly ADX trend filter and volume confirmation.
# Williams Alligator uses three smoothed moving averages (Jaws, Teeth, Lips) to identify trends.
# When the lines are intertwined (no clear trend), we stay out. When they diverge in alignment,
# we follow the direction. Weekly ADX ensures we only trade in strong trends, avoiding whipsaws.
# Volume confirmation adds conviction to trend continuations.
# Designed for very low trade frequency (5-15/year) to minimize fee drag in 1d timeframe.
# Works in bull markets (bullish alignment with price above all lines) and bear markets 
# (bearish alignment with price below all lines).
name = "1d_WilliamsAlligator_WeeklyADX_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    # Get weekly data for ADX filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Williams Alligator (using Smoothed Moving Average - SMMA)
    # Williams Alligator parameters: 
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward  
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    # We use previous values to avoid look-ahead
    
    def smma(data, period):
        """Smoothed Moving Average - equivalent to Wilder's smoothing"""
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: smoothed = (prev * (period-1) + current) / period
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
                else:
                    result[i] = np.nan
        return result
    
    # Calculate SMMA for median price (typical price)
    typical_price = (high + low + close) / 3
    
    # Jaw: 13-period SMMA of typical price, shifted 8 bars
    jaw_raw = smma(typical_price, 13)
    jaw = np.roll(jaw_raw, 8)  # shift forward 8 bars
    jaw[:8] = np.nan  # first 8 values invalid
    
    # Teeth: 8-period SMMA of typical price, shifted 5 bars
    teeth_raw = smma(typical_price, 8)
    teeth = np.roll(teeth_raw, 5)  # shift forward 5 bars
    teeth[:5] = np.nan  # first 5 values invalid
    
    # Lips: 5-period SMMA of typical price, shifted 3 bars
    lips_raw = smma(typical_price, 5)
    lips = np.roll(lips_raw, 3)  # shift forward 3 bars
    lips[:3] = np.nan  # first 3 values invalid
    
    # Align Williams Alligator lines to daily timeframe (already daily data)
    jaw_aligned = jaw  # already aligned to daily
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Calculate weekly ADX for trend strength
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # True Range
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_w[1:] - high_w[:-1]) > (low_w[:-1] - low_w[1:]), 
                       np.maximum(high_w[1:] - high_w[:-1], 0), 0)
    dm_minus = np.where((low_w[:-1] - low_w[1:]) > (high_w[1:] - high_w[:-1]), 
                        np.maximum(low_w[:-1] - low_w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nansum(data[:period]) / period
            # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr_period = 14
    atr = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # DI values
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, atr_period)  # ADX is smoothed DX
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 20-day average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Williams Alligator alignment conditions
        # Bullish alignment: Lips > Teeth > Jaw (green above red above blue)
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Bearish alignment: Lips < Teeth < Jaw (green below red below blue)
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:
            # Long: Bullish alignment AND price above all lines AND strong trend (ADX > 25) AND volume
            price_above_all = (close[i] > lips_aligned[i]) and (close[i] > teeth_aligned[i]) and (close[i] > jaw_aligned[i])
            strong_trend = adx_aligned[i] > 25
            
            if vol_confirm and bullish_alignment and price_above_all and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND price below all lines AND strong trend (ADX > 25) AND volume
            elif (vol_confirm and 
                  bearish_alignment and 
                  (close[i] < lips_aligned[i]) and (close[i] < teeth_aligned[i]) and (close[i] < jaw_aligned[i]) and
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish alignment OR price falls below any line
            price_below_any = (close[i] < lips_aligned[i]) or (close[i] < teeth_aligned[i]) or (close[i] < jaw_aligned[i])
            
            if bearish_alignment or price_below_any:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish alignment OR price rises above any line
            price_above_any = (close[i] > lips_aligned[i]) or (close[i] > teeth_aligned[i]) or (close[i] > jaw_aligned[i])
            
            if bullish_alignment or price_above_any:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals