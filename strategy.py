#!/usr/bin/env python3
"""
1h_Momentum_Volume_Regime_Filter
Hypothesis: Uses 4h trend direction (EMA crossover) and 1d volatility regime (ATR ratio) to filter 1h momentum entries.
Long when: 4h EMA21 > EMA50 (uptrend), 1h RSI > 55, and volume > 1.5x 20-period average.
Short when: 4h EMA21 < EMA50 (downtrend), 1h RSI < 45, and volume > 1.5x 20-period average.
Avoids choppy markets using 1d ATR ratio (ATR10/ATR30 < 0.8) to ensure trending conditions.
Designed for low trade frequency (15-30 trades/year) with volume confirmation to reduce false signals.
"""

name = "1h_Momentum_Volume_Regime_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4H Data for Trend Filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA21 and EMA50 for trend
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === 1D Data for Volatility Regime Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR10 and ATR30 for volatility regime
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # first tr is undefined
    
    atr10_1d = pd.Series(tr1).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr30_1d = pd.Series(tr1).ewm(span=30, adjust=False, min_periods=30).mean().values
    atr_ratio_1d = atr10_1d / atr30_1d  # < 0.8 indicates low volatility (trending)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # === 1H Indicators for Entry Timing ===
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma20 + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 4h EMA50 and 1d ATR30)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (low volatility)
        is_trending = atr_ratio_1d_aligned[i] < 0.8
        
        if position == 0 and is_trending:
            # Long: 4h uptrend, bullish momentum, high volume
            if (ema21_4h_aligned[i] > ema50_4h_aligned[i] and 
                rsi[i] > 55 and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend, bearish momentum, high volume
            elif (ema21_4h_aligned[i] < ema50_4h_aligned[i] and 
                  rsi[i] < 45 and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h trend reversal or momentum fade
            if (ema21_4h_aligned[i] <= ema50_4h_aligned[i] or 
                rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: 4h trend reversal or momentum fade
            if (ema21_4h_aligned[i] >= ema50_4h_aligned[i] or 
                rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals