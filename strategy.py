#!/usr/bin/env python3
"""
1d KAMA Direction + RSI(14) + Chop Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) identifies the dominant trend 
on daily timeframe. RSI(14) filters for momentum strength (avoid overbought/oversold 
chops). Choppiness Index (CHOP) regime filter ensures we only trade in trending 
markets (CHOP < 38.2) and avoid range-bound conditions. Designed for 1d timeframe 
with 30-100 total trades over 4 years (7-25/year) to minimize fee drag while 
capturing sustained moves in both bull and bear markets.
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
    
    # Get weekly data for higher timeframe trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need sufficient weekly data
        return np.zeros(n)
    
    # Calculate 1d KAMA (adaptive trend)
    close_s = pd.Series(close)
    # Efficiency Ratio: |net change| / sum of absolute changes over 10 periods
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants: fastest EMA(2), slowest EMA(30)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close_s.iloc[9]  # seed with first close
    for i in range(10, n):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to 1d (no additional delay needed for EMA-like indicator)
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)  # Using 1w as HTF for stability
    
    # Calculate 1d RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 1d Choppiness Index (CHOP) - 14 period
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = tr1[0]
    # Sum of TR over 14 periods
    atr_sum = np.full(n, np.nan)
    for i in range(13, n):
        atr_sum[i] = np.sum(tr[i-13:i+1])
    # Highest high and lowest low over 14 periods
    hh = np.full(n, np.nan)
    ll = np.full(n, np.nan)
    for i in range(13, n):
        hh[i] = np.max(high[i-13:i+1])
        ll[i] = np.min(low[i-13:i+1])
    # CHOP = 100 * log10( sum(TR) / (HH - LL) ) / log10(14)
    chop = np.full(n, np.nan)
    for i in range(13, n):
        if hh[i] > ll[i]:  # avoid division by zero
            chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 100  # max choppy when range is zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for KAMA seed, RSI, and CHOP
    start_idx = max(20, 14, 13)  # KAMA needs 10, RSI 14, CHOP 14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_values[i]
        chop_val = chop[i]
        
        # Trend filter: price relative to KAMA
        uptrend = curr_close > kama_val
        downtrend = curr_close < kama_val
        
        # Momentum filter: RSI not extreme (avoid choppy reversals)
        rsi_ok = (rsi_val > 30) and (rsi_val < 70)
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        trending_regime = chop_val < 38.2
        
        if position == 0:
            # Enter long: price above KAMA + good momentum + trending regime
            long_entry = uptrend and rsi_ok and trending_regime
            # Enter short: price below KAMA + good momentum + trending regime
            short_entry = downtrend and rsi_ok and trending_regime
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price crosses below KAMA OR regime turns choppy
            if curr_close < kama_val or chop_val >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA OR regime turns choppy
            if curr_close > kama_val or chop_val >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0