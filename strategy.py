#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour MACD histogram with 1-day ADX trend filter and volume confirmation.
# Long when: MACD histogram > 0 and rising, daily ADX > 25 (trending), volume > 1.8x 20-period average
# Short when: MACD histogram < 0 and falling, daily ADX > 25 (trending), volume > 1.8x 20-period average
# Exit when: MACD histogram crosses zero (opposite sign)
# MACD captures momentum, ADX filters for trending markets only, volume confirms strength.
# Works in bull (buy strength) and bear (sell weakness) by only trading in trending regimes.
# Target: 20-30 trades/year per symbol.
name = "6h_MACD_ADX_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate MACD (12,26,9)
    close_series = pd.Series(close)
    ema12 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close_series.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd_line - signal_line
    macd_hist = macd_hist.values
    
    # Daily ADX for trend strength
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    atr = smooth(tr, 14)
    di_plus = 100 * smooth(dm_plus, 14) / atr
    di_minus = 100 * smooth(dm_minus, 14) / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = smooth(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 20-period volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # Wait for MACD and volume calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(macd_hist[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hist = macd_hist[i]
        hist_prev = macd_hist[i-1]
        adx_val = adx_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: MACD histogram > 0 and rising, ADX > 25, volume spike
            if (hist > 0 and hist > hist_prev and 
                adx_val > 25 and vol > 1.8 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: MACD histogram < 0 and falling, ADX > 25, volume spike
            elif (hist < 0 and hist < hist_prev and 
                  adx_val > 25 and vol > 1.8 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: MACD histogram crosses below zero
            if hist < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: MACD histogram crosses above zero
            if hist > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals