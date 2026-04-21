#!/usr/bin/env python3
"""
4h_HTF_RSI_Choppiness_VolumeBreakout_V1
Hypothesis: Use 1d RSI(14) > 50 for bull trend, < 50 for bear trend. Enter on 4h Donchian(20) breakout in trend direction with volume > 2.0x 20-bar MA. Exit on ATR(14) stoploss (2.5x) or opposite Donchian breakout. Choppiness Index(14) > 61.8 avoids ranging markets. Discrete position sizing 0.25 minimizes fee churn. Target 30-60 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for RSI trend filter
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d RSI(14) for Trend Filter ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[rs == 0] = 100  # when avg_loss is 0
    
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === 4h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    # Volume MA (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14-period)
    def choppiness_index(high, low, close, window=14):
        atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        hh = pd.Series(high).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low).rolling(window=window, min_periods=window).min().values
        chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(window)
        # Avoid division by zero and invalid values
        chop[(hh - ll) == 0] = 50
        chop[np.isnan(chop)] = 50
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # strong volume confirmation
        chop_ok = chop[i] > 61.8  # ranging market filter (avoid chop)
        rsi_bull = rsi_aligned[i] > 50  # 1d bull trend
        rsi_bear = rsi_aligned[i] < 50  # 1d bear trend
        
        if position == 0:
            # Long: Donchian breakout above with volume, bull trend, not choppy
            if price > donchian_high[i] and vol_ok and rsi_bull and not chop_ok:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below with volume, bear trend, not choppy
            elif price < donchian_low[i] and vol_ok and rsi_bear and not chop_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite Donchian breakout
            if price < close[i-1] - 2.5 * atr[i] or price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite Donchian breakout
            if price > close[i-1] + 2.5 * atr[i] or price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_RSI_Choppiness_VolumeBreakout_V1"
timeframe = "4h"
leverage = 1.0