#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_and_ChopFilter
Hypothesis: On 1d timeframe, Kaufman Adaptive Moving Average (KAMA) identifies the adaptive trend direction, RSI(14) filters for momentum exhaustion (avoid buying overbought/selling oversold), and Choppiness Index (CHOP) regime filter avoids whipsaws in ranging markets. This combination captures medium-term trend continuations with controlled trade frequency (target: 7-25 trades/year). Works in bull markets via KAMA uptrend + RSI < 70, and in bear markets via KAMA downtrend + RSI > 30. Uses 1w EMA34 as higher timeframe trend confirmation to avoid counter-trend trades.
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
    
    # Get 1d data for indicators (primary timeframe is 1d, so this is LTF data)
    # But we need to ensure we're using the correct data - since timeframe=1d, prices IS 1d data
    # However, for safety and MTF compliance, we'll treat prices as 1d and get 1w for HTF
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w for HTF trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate KAMA on 1d (primary timeframe)
    # KAMA parameters: ER period=10, fast=2, slow=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))).reshape(-1, 1)  # placeholder, will compute properly below
    
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        if i >= 10:
            direction = np.abs(close[i] - close[i-10])
            volatility_sum = np.sum(np.abs(np.diff(close[i-10:i+1])))
            if volatility_sum > 0:
                er[i] = direction / volatility_sum
            else:
                er[i] = 0
    er[:10] = 0
    
    # Smoothing constants
    fast_sc = 2/(2+1)
    slow_sc = 2/(30+1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # neutral before warmup
    
    # Calculate Choppiness Index on 1d (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high) - min(low))))
    atr_1d = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
    atr_1d = np.concatenate([[np.nan], atr_1d])
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop_raw = 100 * np.log10(atr_sum / (np.log10(14) * (max_high - min_low)))
    chop = np.where((max_high - min_low) > 0, chop_raw, 50)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(34, 14, 10)  # EMA34, RSI, KAMA ER period
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get values
        ema_val = ema_34_1w_aligned[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        close_val = close[i]
        
        # Regime filter: CHOP > 50 indicates ranging market (avoid breakouts in strong trends)
        # But for trend following, we want CHOP < 50 to indicate trending market
        trending_regime = chop_val < 50
        
        if position == 0:
            # Look for entry signals: KAMA trend alignment with RSI filter and HTF trend
            # Long: price > KAMA (uptrend), RSI < 70 (not overbought), HTF uptrend (close > EMA34_1w), trending regime
            long_signal = (close_val > kama_val) and (rsi_val < 70) and (close_val > ema_val) and trending_regime
            # Short: price < KAMA (downtrend), RSI > 30 (not oversold), HTF downtrend (close < EMA34_1w), trending regime
            short_signal = (close_val < kama_val) and (rsi_val > 30) and (close_val < ema_val) and trending_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Trend reversal: price < KAMA (downtrend)
            # 2. Overbought: RSI > 70
            # 3. HTF trend reversal: close < EMA34_1w
            if (close_val < kama_val) or (rsi_val > 70) or (close_val < ema_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Trend reversal: price > KAMA (uptrend)
            # 2. Oversold: RSI < 30
            # 3. HTF trend reversal: close > EMA34_1w
            if (close_val > kama_val) or (rsi_val < 30) or (close_val > ema_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_RSI_and_ChopFilter"
timeframe = "1d"
leverage = 1.0