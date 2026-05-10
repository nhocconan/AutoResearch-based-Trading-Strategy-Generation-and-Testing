# 1D_KAMA_Trend_RSI_Chop_v1
# Hypothesis: Daily KAMA trend direction combined with RSI overbought/oversold and Choppiness regime filter.
# Works in bull markets by following KAMA trend, in bear markets by fading extremes in choppy regimes.
# Uses daily timeframe to reduce trade frequency (target: 10-25 trades/year) and avoid fee drag.
# KAMA adapts to market noise, making it effective in both trending and ranging conditions.

#!/usr/bin/env python3

name = "1D_KAMA_Trend_RSI_Chop_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA(10) on daily close
    close_series = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_series.diff(10))
    volatility = abs(close_series.diff(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.finfo(float).eps)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) on daily close
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate Choppiness Index(14)
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros(n)
    for i in range(14, n):
        if atr[i] > 0 and highest_high[i] > lowest_low[i]:
            sum_atr = pd.Series(atr[i-13:i+1]).sum()
            chop[i] = 100 * np.log10(sum_atr / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # Align weekly EMA to daily
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = close[i] > ema_1w_aligned[i]
        price_below_ema = close[i] < ema_1w_aligned[i]
        
        # KAMA direction
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # RSI extremes
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        
        # Chop regime: chop > 61.8 = range (mean revert), chop < 38.2 = trending (trend follow)
        chop_range = chop[i] > 61.8
        chop_trending = chop[i] < 38.2
        
        if position == 0:
            # Long entry: KAMA up + price above weekly EMA + (trending OR oversold in range)
            if (kama_up and price_above_ema and 
                (chop_trending or (chop_range and rsi_oversold))):
                signals[i] = 0.25
                position = 1
            # Short entry: KAMA down + price below weekly EMA + (trending OR overbought in range)
            elif (kama_down and price_below_ema and 
                  (chop_trending or (chop_range and rsi_overbought))):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA down OR price below weekly EMA OR overbought in trending market
            if (not kama_up or not price_above_ema or 
                (chop_trending and rsi_overbought)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA up OR price above weekly EMA OR oversold in trending market
            if (not kama_down or not price_below_ema or 
                (chop_trending and rsi_oversold)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals