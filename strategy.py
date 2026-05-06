#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R for overbought/oversold conditions and 1w ADX for trend strength
# - Enters long when 1d Williams %R < -80 (oversold) and 1w ADX > 25 (trending) with price above 1d VWAP
# - Enters short when 1d Williams %R > -20 (overbought) and 1w ADX > 25 (trending) with price below 1d VWAP
# - Uses 6-hour RSI(14) for entry timing: long when RSI crosses above 30, short when RSI crosses below 70
# - Exits when Williams %R returns to neutral range (-50) or ADX weakens (< 20)
# - Designed to catch mean reversion within strong trends on higher timeframes
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_1dWilliamsR_1wADX_TrendMeanRev"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Get 1w data for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d VWAP
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_numerator = np.cumsum(typical_price_1d * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    # Handle division by zero
    vwap = np.where(vwap_denominator == 0, typical_price_1d, vwap)
    
    # Calculate 1w ADX (14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_1w
    di_minus = 100 * dm_minus_smooth / atr_1w
    # Handle division by zero
    di_plus = np.where(atr_1w == 0, 0, di_plus)
    di_minus = np.where(atr_1w == 0, 0, di_minus)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = wilders_smoothing(dx, 14)
    
    # Calculate 6h RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1d indicators to 6h timeframe
    williams_r_6h = align_htf_to_ltf(prices, df_1d, williams_r)
    vwap_6h = align_htf_to_ltf(prices, df_1d, vwap)
    
    # Align 1w ADX to 6h timeframe
    adx_6h = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(williams_r_6h[i]) or np.isnan(vwap_6h[i]) or np.isnan(adx_6h[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Strong trend condition (ADX > 25)
            strong_trend = adx_6h[i] > 25
            
            if strong_trend:
                # Long: oversold (Williams %R < -80) and price above VWAP, RSI crossing above 30
                if (williams_r_6h[i] < -80 and 
                    close[i] > vwap_6h[i] and 
                    rsi[i] > 30 and rsi[i-1] <= 30):
                    signals[i] = 0.25
                    position = 1
                # Short: overbought (Williams %R > -20) and price below VWAP, RSI crossing below 70
                elif (williams_r_6h[i] > -20 and 
                      close[i] < vwap_6h[i] and 
                      rsi[i] < 70 and rsi[i-1] >= 70):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral (-50) or trend weakens (ADX < 20)
            if williams_r_6h[i] >= -50 or adx_6h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral (-50) or trend weakens (ADX < 20)
            if williams_r_6h[i] <= -50 or adx_6h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals