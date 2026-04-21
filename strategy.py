#!/usr/bin/env python3
"""
1d_Keltner_Channel_Breakout_WeeklyTrend_VolumeSpike_v1
Hypothesis: Daily price breaks above/below Keltner Channel (EMA20 ± 2*ATR10) 
filtered by weekly EMA34 trend and volume spike (>2x 20-day average). 
Keltner channels adapt to volatility, reducing false breakouts in ranging markets. 
Weekly trend filter ensures we only trade in the direction of the higher timeframe momentum. 
Volume spike confirms institutional participation. Designed for low trade frequency 
(10-25/year) to minimize fee drag while capturing strong trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === Daily OHLC for Keltner Channel calculation ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Keltner Channel: EMA20 ± 2*ATR10 ===
    # EMA20 of close
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range and ATR(10)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.rolling(window=10, min_periods=10).mean().values
    
    # Keltner Bands
    keltner_upper = ema_20 + 2.0 * atr_10
    keltner_lower = ema_20 - 2.0 * atr_10
    
    # === Weekly EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Volume filter: >2x 20-day average ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) 
            or np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Keltner Upper, weekly uptrend, volume spike
            if price > keltner_upper[i] and price > ema_34_1w_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Keltner Lower, weekly downtrend, volume spike
            elif price < keltner_lower[i] and price < ema_34_1w_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit long: price closes below EMA20 (mean reversion) or weekly trend turns bearish
            if price < ema_20[i] or price < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above EMA20 or weekly trend turns bullish
            if price > ema_20[i] or price > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Keltner_Channel_Breakout_WeeklyTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0