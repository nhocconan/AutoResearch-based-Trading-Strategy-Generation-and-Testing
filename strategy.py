#!/usr/bin/env python3
"""
12h_HTF_Donchian20_Breakout_Volume_ATRFilter_V1
Hypothesis: Use 1d Donchian(20) breakout with volume confirmation (>1.5x 20-bar volume MA) and 1w trend filter (price above/below 50-week EMA). Enter on break of prior 20-bar high/low on 12h chart with volume confirmation. Exit on ATR stoploss (2.0x) or opposite signal. Donchian channels provide clear structure, volume confirms legitimacy, weekly EMA filters counter-trend noise. Target 12-37 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')   # for Donchian and volume MA
    df_1w = get_htf_data(prices, '1w')   # for weekly EMA trend filter
    
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d Donchian Channel (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper: 20-period high
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower: 20-period low
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1d volume MA (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # === 1w EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 12h Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma_1d_aligned[i]  # volume confirmation
        trend_up = ema_50_1w_aligned[i] < price   # price above weekly EMA50 = uptrend
        trend_down = ema_50_1w_aligned[i] > price # price below weekly EMA50 = downtrend
        
        # Entry conditions: break of prior 20-bar high/low (Donchian)
        if i >= 20:
            prior_high = np.max(high[i-20:i])
            prior_low = np.min(low[i-20:i])
        else:
            prior_high = high[i]
            prior_low = low[i]
        
        if position == 0:
            # Long: break above prior 20-bar high with volume and uptrend
            if price > prior_high and vol_ok and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: break below prior 20-bar low with volume and downtrend
            elif price < prior_low and vol_ok and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal
            if price < close[i-1] - 2.0 * atr[i] or (price < prior_low and vol_ok and trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal
            if price > close[i-1] + 2.0 * atr[i] or (price > prior_high and vol_ok and trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_HTF_Donchian20_Breakout_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0