#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Donchian breakout with 1-week ATR filter and volume confirmation
# Long when price breaks above daily Donchian high with ATR(1w) > 1.5*ATR(12w) and volume > 1.5x average
# Short when price breaks below daily Donchian low with ATR(1w) > 1.5*ATR(12w) and volume > 1.5x average
# Daily Donchian provides strong trend structure, ATR filter ensures volatility expansion,
# Volume confirms breakout strength. Works in bull/bear markets by capturing genuine breakouts.
# Target: 12-37 trades per year (48-148 over 4 years) with 0.25 position sizing.

name = "12h_1dDonchian_1wATR_Volume_Breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Donchian channels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period high/low)
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1-week ATR for volatility filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 12:
        return np.zeros(n)
    
    # True Range calculation for 1-week ATR
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr.rolling(window=12, min_periods=12).mean().values
    
    # Calculate 12-week ATR for volatility ratio
    tr_12w = tr.rolling(window=12, min_periods=12).mean().values
    
    # Align ATR values to 12h timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_12w_aligned = align_htf_to_ltf(prices, df_1w, tr_12w)
    
    # Volatility filter: ATR(1w) > 1.5 * ATR(12w) indicating volatility expansion
    vol_filter = atr_1w_aligned > (1.5 * atr_12w_aligned)
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or np.isnan(atr_12w_aligned[i]) or
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above daily Donchian high with volatility and volume confirmation
            if close[i] > donchian_high_aligned[i] and vol_filter[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below daily Donchian low with volatility and volume confirmation
            elif close[i] < donchian_low_aligned[i] and vol_filter[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below daily Donchian low (trend reversal)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above daily Donchian high (trend reversal)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals