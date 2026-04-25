#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Regime Filter
Hypothesis: On daily timeframe, KAMA (Kaufman Adaptive Moving Average) captures the dominant trend with low lag in ranging markets.
RSI(14) provides momentum confirmation, while Choppiness Index (CHOP) filters regimes: 
- CHOP > 61.8 = ranging (mean revert at RSI extremes)
- CHOP < 38.2 = trending (follow KAMA direction)
This strategy works in both bull and bear markets by adapting to regime: 
In trending regimes, we follow KAMA breakouts; in ranging regimes, we fade RSI extremes.
Designed for 1d timeframe with tight entry conditions to achieve 7-25 trades/year.
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
    
    # Get 1w data for higher timeframe trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on daily close (using 1d data from prices since timeframe=1d)
    # Efficiency ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    close_s = pd.Series(close)
    change = abs(close_s - close_s.shift(10))
    volatility = abs(close_s - close_s.shift(1)).rolling(window=10, min_periods=10).sum()
    ER = change / volatility.replace(0, np.nan)
    # Smoothing constants: fastest SC = 2/(2+1)=0.667, slowest SC = 2/(30+1)=0.0645
    SC = (ER * 0.603 + 0.0645) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close_s.iloc[9]  # seed after 10 periods
    for i in range(10, n):
        if not np.isnan(SC.iloc[i]):
            kama[i] = kama[i-1] + SC.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate Choppiness Index (CHOP) on daily data
    # True Range = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    # Sum of TR over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # CHOP = 100 * log10(atr_sum / (hh - ll)) / log10(14)
    range_hl = hh - ll
    # Avoid division by zero
    chop = np.full(n, np.nan)
    mask = (range_hl > 0) & (~np.isnan(atr_sum))
    chop[mask] = 100 * np.log10(atr_sum[mask] / range_hl[mask]) / np.log10(14)
    
    # Get 1w EMA50 for higher timeframe trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_1w_trend = ema_50_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Regime filter: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
            if chop_val > 61.8:
                # Ranging market: mean reversion at RSI extremes
                long_entry = (rsi_val < 30) and (curr_close > kama_val)  # oversold and price above KAMA
                short_entry = (rsi_val > 70) and (curr_close < kama_val)  # overbought and price below KAMA
            else:
                # Trending market: follow KAMA direction with 1w EMA filter
                long_entry = (curr_close > kama_val) and (curr_close > ema_1w_trend)
                short_entry = (curr_close < kama_val) and (curr_close < ema_1w_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below KAMA OR RSI > 70 (overbought) OR regime shifts to strong ranging with RSI > 60
            if (curr_close < kama_val) or (rsi_val > 70) or (chop_val > 61.8 and rsi_val > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above KAMA OR RSI < 30 (oversold) OR regime shifts to strong ranging with RSI < 40
            if (curr_close > kama_val) or (rsi_val < 30) or (chop_val > 61.8 and rsi_val < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopRegime"
timeframe = "1d"
leverage = 1.0