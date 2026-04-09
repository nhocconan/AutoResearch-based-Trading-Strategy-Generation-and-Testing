#!/usr/bin/env python3
# 6h_adx_williams_alligator_v1
# Hypothesis: 6h strategy combining ADX trend strength with Williams Alligator (3 SMAs) to filter false breakouts.
# Long when: ADX > 25 (strong trend) + price > Alligator Jaw (13-period SMA shifted 8) + price > Alligator Teeth (8-period SMA shifted 5)
# Short when: ADX > 25 + price < Alligator Jaw + price < Alligator Teeth
# Exit when: ADX < 20 (trend weakens) OR price crosses back inside Alligator Mouth (between Teeth and Lips)
# Uses 12h timeframe for HTF trend confirmation: only take longs when 12h close > 12h EMA50, shorts when < 12h EMA50
# Discrete position sizing: 0.25 to minimize fee churn
# Target: 12-25 trades/year (50-100 total over 4 years) per symbol

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_williams_alligator_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Williams Alligator (6h) ===
    # Jaw: 13-period SMA, shifted 8 bars
    # Teeth: 8-period SMA, shifted 5 bars  
    # Lips: 5-period SMA, shifted 3 bars
    close_s = pd.Series(close)
    jaw = close_s.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = close_s.rolling(window=8, min_periods=8).mean().shift(5).values
    lips = close_s.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # === ADX (6h) ===
    # +DM, -DM, TR
    plus_dm = high[1:] - high[:-1]
    minus_dm = low[:-1] - low[1:]
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Pad arrays to match original length (shifted by 1 due to diff)
    plus_dm_padded = np.concatenate([[np.nan], plus_dm])
    minus_dm_padded = np.concatenate([[np.nan], minus_dm])
    tr_padded = np.concatenate([[np.nan], tr])
    
    atr_period = 14
    atr = wilders_smoothing(tr_padded, atr_period)
    plus_di = 100 * wilders_smoothing(plus_dm_padded, atr_period) / (atr + 1e-10)
    minus_di = 100 * wilders_smoothing(minus_dm_padded, atr_period) / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilders_smoothing(dx, atr_period)
    
    # === 12h HTF Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    close_12h_s = pd.Series(close_12h)
    ema50_12h = close_12h_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ADX < 20 (trend weakens) OR price crosses below Teeth (mouth)
            if adx[i] < 20.0 or close[i] < teeth[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ADX < 20 (trend weakens) OR price crosses above Teeth (mouth)
            if adx[i] < 20.0 or close[i] > teeth[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for entry with ADX trend strength and 12h alignment
            strong_trend = adx[i] > 25.0
            bullish_alligator = (close[i] > jaw[i]) and (close[i] > teeth[i])
            bearish_alligator = (close[i] < jaw[i]) and (close[i] < teeth[i])
            htf_bullish = close[i] > ema50_12h_aligned[i]
            htf_bearish = close[i] < ema50_12h_aligned[i]
            
            if strong_trend and bullish_alligator and htf_bullish:
                position = 1
                signals[i] = 0.25
            elif strong_trend and bearish_alligator and htf_bearish:
                position = -1
                signals[i] = -0.25
    
    return signals