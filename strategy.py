#!/usr/bin/env python3
# 1d_kama_rsi_chop_v2
# Hypothesis: Daily KAMA trend direction + RSI mean reversion + Choppiness regime filter.
# In trending markets (CHOP < 38.2): enter pullbacks in KAMA direction.
# In ranging markets (CHOP > 61.8): fade extreme RSI at Bollinger Bands.
# Designed for low trade frequency (<25/year) to avoid fee drag. Works in bull/bear via regime adaptation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v2"
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
    
    # Weekly HTF data for Donchian trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian channels (20-period)
    highest_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high = align_htf_to_ltf(prices, df_1w, highest_20)
    donchian_low = align_htf_to_ltf(prices, df_1w, lowest_20)
    
    # Daily indicators
    # KAMA (10,2,30)
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / (volatility + 1e-10)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [close[0]]  # seed
    for i in range(1, n):
        kama.append(kama[-1] + sc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # RSI (14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20,2)
    sma20 = close_s.rolling(window=20, min_periods=20).mean().values
    std20 = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # Choppiness Index (14)
    atr1 = abs(high - low)
    atr2 = abs(high - np.roll(close, 1))
    atr3 = abs(low - np.roll(close, 1))
    atr1[0] = atr2[0] = atr3[0] = 0
    tr = np.maximum(np.maximum(atr1, atr2), atr3)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(upper_bb[i]) or
            np.isnan(lower_bb[i]) or np.isnan(chop[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches upper Bollinger Band OR Donchian breakout fails
            if close[i] >= upper_bb[i] or close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches lower Bollinger Band OR Donchian breakout fails
            if close[i] <= lower_bb[i] or close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation: current volume > 1.3x 20-period average
            volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            volume_confirmed = volume[i] > 1.3 * volume_ma[i] if not np.isnan(volume_ma[i]) else False
            
            if volume_confirmed:
                if chop[i] < 38.2:  # Trending regime
                    # Long: pullback to KAMA in uptrend
                    if close[i] > kama[i] and close[i] < kama[i] * 1.02 and close[i] > donchian_low[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short: pullback to KAMA in downtrend
                    elif close[i] < kama[i] and close[i] > kama[i] * 0.98 and close[i] < donchian_high[i]:
                        position = -1
                        signals[i] = -0.25
                elif chop[i] > 61.8:  # Ranging regime
                    # Long: RSI oversold near lower BB
                    if rsi[i] < 30 and close[i] <= lower_bb[i] * 1.01:
                        position = 1
                        signals[i] = 0.25
                    # Short: RSI overbought near upper BB
                    elif rsi[i] > 70 and close[i] >= upper_bb[i] * 0.99:
                        position = -1
                        signals[i] = -0.25
    
    return signals