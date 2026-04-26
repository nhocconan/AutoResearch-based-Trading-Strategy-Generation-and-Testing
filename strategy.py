#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v2
Hypothesis: 1d KAMA trend direction combined with RSI extremes and Choppiness Index regime filter captures sustainable momentum while avoiding choppy markets. Works in both bull and bear regimes by adapting to market structure via Chop filter. Targets 30-100 trades over 4 years (7-25/year) with discrete position sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF trend context
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for weekly trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate KAMA (1d ER=10, fast=2, slow=30) - adaptive trend
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(10, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0).clip(0, 1)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    kama_aligned = kama  # already 1d aligned
    
    # Calculate RSI(14) for momentum extremes
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Calculate Choppiness Index(14) for regime filter
    atr_1 = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    atr_1 = np.insert(atr_1, 0, atr_1[0] if len(atr_1) > 0 else 0)
    atr_sum = pd.Series(atr_1).rolling(14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(34, 14, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend filter (EMA34)
        uptrend_1w = close[i] > ema_34_1w_aligned[i]
        downtrend_1w = close[i] < ema_34_1w_aligned[i]
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI extremes: oversold <30, overbought >70
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Chop regime: trending when CHOP < 38.2, ranging when CHOP > 61.8
        chop_trending = chop[i] < 38.2
        chop_ranging = chop[i] > 61.8
        
        # Long logic: KAMA uptrend + RSI oversold + weekly uptrend + trending regime
        if price_above_kama and rsi_oversold and uptrend_1w and chop_trending:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: KAMA downtrend + RSI overbought + weekly downtrend + trending regime
        elif price_below_kama and rsi_overbought and downtrend_1w and chop_trending:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: opposite signal or regime change to ranging
        elif position == 1 and (price_below_kama or rsi_overbought or not chop_trending):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (price_above_kama or rsi_oversold or not chop_trending):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v2"
timeframe = "1d"
leverage = 1.0