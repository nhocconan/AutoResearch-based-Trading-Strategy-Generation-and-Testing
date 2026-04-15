#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d RSI mean reversion + volume confirmation
# Uses Choppiness Index (14) on 4h to detect ranging markets (CHOP > 61.8) and trending markets (CHOP < 38.2).
# In ranging markets: long when 1d RSI < 30, short when 1d RSI > 70 with volume confirmation.
# In trending markets: follow 1d EMA(50) direction with 4h Donchian breakout for entry timing.
# Designed to work in both bull and bear markets by adapting to market regime.
# Target: 20-60 total trades over 4 years (5-15/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for Choppiness Index calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate True Range and ATR(14) for Choppiness Index
    tr1 = np.abs(high_4h[1:] - low_4h[:-1])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.max([high_4h[0] - low_4h[0], 0])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index: 100 * log10(sum(ATR)/ (max(high)-min(low)) ) / log10(14)
    max_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(pd.Series(atr).rolling(window=14, min_periods=14).sum().values / (max_high - min_low)) / np.log10(14)
    
    # Load 1d data for RSI and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 14-period RSI on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 50-period EMA on 1d
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Load 4h data for Donchian breakout (used in trending markets)
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align all indicators to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i])):
            continue
        
        chop_val = chop_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        ema_val = ema_50_1d_aligned[i]
        
        # Regime detection: Choppiness Index
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        if is_ranging:
            # Mean reversion in ranging markets
            if (rsi_val < 30 and volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
                position <= 0):
                position = 1
                signals[i] = base_size
            elif (rsi_val > 70 and volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
                  position >= 0):
                position = -1
                signals[i] = -base_size
            # Exit on RSI crossing 50
            elif position == 1 and rsi_val > 50:
                position = 0
                signals[i] = 0.0
            elif position == -1 and rsi_val < 50:
                position = 0
                signals[i] = 0.0
        elif is_trending:
            # Trend following in trending markets
            if (close[i] > donch_high_aligned[i] and close[i] > ema_val and
                position <= 0):
                position = 1
                signals[i] = base_size
            elif (close[i] < donch_low_aligned[i] and close[i] < ema_val and
                  position >= 0):
                position = -1
                signals[i] = -base_size
            # Exit on reverse Donchian breakout
            elif position == 1 and close[i] < donch_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > donch_high_aligned[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Chop_RSI_EMA_Regime"
timeframe = "4h"
leverage = 1.0