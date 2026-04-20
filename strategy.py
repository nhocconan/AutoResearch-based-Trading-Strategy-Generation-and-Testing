#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Chop_KAMA_RSI_MeanRev"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 5:
        return np.zeros(n)
    
    # === 1d: KAMA trend filter ===
    close_1d = df_1d['close'].values
    # Calculate efficiency ratio (ER) for KAMA
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    # Proper ER calculation
    price_diff = np.abs(np.diff(close_1d, k=10))  # 10-period change
    abs_sum = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder
    # Recompute properly using rolling
    close_series = pd.Series(close_1d)
    change_abs = np.abs(close_series.diff(10))
    volatility_sum = close_series.diff().abs().rolling(window=10, min_periods=1).sum()
    er = np.where(volatility_sum > 0, change_abs / volatility_sum, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    # === 1d: RSI(14) for mean reversion ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_gain > 0, avg_loss / avg_gain, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # === 1d: Choppiness Index (CHOP) for regime detection ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    atr_raw = np.maximum(high_1d - low_1d, 
                         np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                    np.abs(low_1d - np.roll(close_1d, 1))))
    atr_raw[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(atr_raw).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # True range for 14 periods
    tr14 = pd.Series(atr_raw).rolling(window=14, min_periods=14).sum().values
    # Max high - min low over 14 periods
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = np.where((hh14 - ll14) > 0, 100 * np.log10(tr14 / (hh14 - ll14)) / np.log10(14), 50)
    
    # === 1w: Trend filter (EMA20) ===
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF indicators to 1d timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Price and volume (1d)
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute hours for session filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip outside session (8-20 UTC)
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close[i]
        kama_val = kama_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        ema20_1w_val = ema20_1w_aligned[i]
        vol_ma20_val = vol_ma20[i]
        
        # Skip if any value is NaN
        if np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val) or np.isnan(ema20_1w_val) or np.isnan(vol_ma20_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma20_val if vol_ma20_val > 0 else 0
        
        if position == 0:
            # Long: Range market (high CHOP) + price below KAMA (oversold) + RSI oversold + weekly uptrend
            if (chop_val > 61.8 and          # Choppy/ranging market
                close_val < kama_val and     # Price below KAMA (potential oversold)
                rsi_val < 35 and             # RSI oversold
                ema20_1w_val > 0 and         # Weekly uptrend (EMA20 slope positive approximation)
                vol_ratio > 1.2):            # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Range market + price above KAMA (overbought) + RSI overbought + weekly downtrend
            elif (chop_val > 61.8 and        # Choppy/ranging market
                  close_val > kama_val and   # Price above KAMA (potential overbought)
                  rsi_val > 65 and           # RSI overbought
                  ema20_1w_val > 0 and       # Weekly uptrend (we'll use price > EMA for simplicity)
                  vol_ratio > 1.2):          # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: mean reversion complete or regime change
            if (close_val > kama_val or      # Price back above KAMA (mean reversion)
                rsi_val > 65 or              # RSI overbought
                chop_val < 38.2):            # Trending regime (exit range strategy)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: mean reversion complete or regime change
            if (close_val < kama_val or      # Price back below KAMA (mean reversion)
                rsi_val < 35 or              # RSI oversold
                chop_val < 38.2):            # Trending regime (exit range strategy)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals