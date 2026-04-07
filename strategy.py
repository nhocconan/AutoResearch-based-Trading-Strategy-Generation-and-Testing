#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with 1-day volatility filter and weekly trend
# Long when price breaks above 20-period Donchian high (1d) + 1d ATR ratio > 1.2 (expanding volatility) + 1w close > 1w EMA50 (bullish trend)
# Short when price breaks below 20-period Donchian low (1d) + 1d ATR ratio > 1.2 + 1w close < 1w EMA50 (bearish trend)
# Exit when price returns to Donchian midpoint or ATR ratio < 0.8 (low volatility)
# Stoploss at 2.5 * ATR(14) from entry
# Position size: 0.25 (25% of capital)
# Target: 75-150 total trades over 4 years (19-38/year)

name = "12h_donchian20_1d_atr_ratio_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day Donchian channels (20 periods)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_1d_s = pd.Series(high_1d)
    low_1d_s = pd.Series(low_1d)
    donchian_high = high_1d_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1d_s.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1-day ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close, 1))  # close is 12h, but we use it for TR calc approximation
    tr3 = np.abs(low_1d - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_ma = pd.Series(atr_1d).ewm(span=30, adjust=False, min_periods=30).mean().values  # 30-period average ATR
    atr_ratio = atr_1d / (atr_1d_ma + 1e-10)  # Current ATR vs average ATR
    
    # Align 1-day data to 12h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-week EMA(50)
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema_50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12-period ATR(14) for stoploss
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.roll(close, 1))
    tr3_12h = np.abs(low - np.roll(close, 1))
    tr2_12h[0] = tr1_12h[0]
    tr3_12h[0] = tr1_12h[0]
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_12h[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to Donchian midpoint or low volatility
            elif close[i] >= donchian_mid_aligned[i] or atr_ratio_aligned[i] < 0.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to Donchian midpoint or low volatility
            elif close[i] <= donchian_mid_aligned[i] or atr_ratio_aligned[i] < 0.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volatility expansion and weekly trend
            volatility_expansion = atr_ratio_aligned[i] > 1.2
            
            # Long: price breaks above Donchian high + volatility expansion + bullish weekly trend
            if close[i] > donchian_high_aligned[i] and volatility_expansion and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volatility expansion + bearish weekly trend
            elif close[i] < donchian_low_aligned[i] and volatility_expansion and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals