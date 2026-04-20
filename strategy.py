#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with Volume Confirmation and ATR Stop
# - Long when price breaks above 4h Donchian upper band (20-period high) with volume > 1.5x 20-period average
# - Short when price breaks below 4h Donchian lower band (20-period low) with volume > 1.5x 20-period average
# - Exit when price crosses back through Donchian midline or ATR-based stop hit
# - Uses 1d trend filter: only take long if 1d EMA50 > EMA200, short if EMA50 < EMA200
# - Target: 25-40 trades per year per symbol (100-160 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMAs for trend filter
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1d = close_series_1d.ewm(span=200, min_periods=200, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate ATR for stop loss (using 1d data)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(atr_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: 1d EMA50 > EMA200 for long, EMA50 < EMA200 for short
        bullish_trend = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        bearish_trend = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian upper + volume surge + bullish trend
            if price > donchian_upper[i] and vol > 1.5 * vol_ma[i] and bullish_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below Donchian lower + volume surge + bearish trend
            elif price < donchian_lower[i] and vol > 1.5 * vol_ma[i] and bearish_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below Donchian midline OR ATR stop hit (2*ATR)
            if price < donchian_mid[i] or price < entry_price - 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midline OR ATR stop hit (2*ATR)
            if price > donchian_mid[i] or price > entry_price + 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_ATRStop"
timeframe = "4h"
leverage = 1.0