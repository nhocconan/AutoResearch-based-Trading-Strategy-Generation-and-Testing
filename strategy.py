#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 4h RSI mean reversion with 200 EMA filter.
# Uses daily Choppiness Index to filter regimes: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending (avoid).
# In range regime, RSI < 30 = long, RSI > 70 = short, only when price > EMA200 (bull bias) or price < EMA200 (bear bias).
# Designed to work in both bull and bear markets by avoiding trending regimes and fading extremes in ranges.
name = "4h_ChopRSI_MeanReversion_EMA200"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for Choppiness Index (requires daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Daily Choppiness Index (14-period)
    # True Range = max(high-low, abs(high-previous close), abs(low-previous close))
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr.values).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    sum_tr14 = pd.Series(tr.values).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh14 = df_1d['high'].rolling(window=14, min_periods=14).max().values
    ll14 = df_1d['low'].rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index formula: 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
    # Avoid division by zero
    range14 = hh14 - ll14
    chop = np.where(range14 > 0, 100 * np.log10(sum_tr14 / range14) / np.log10(14), 50)
    
    # Align Choppiness Index to 4h timeframe
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA200 trend filter
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need enough data for EMA200
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(chop_4h[i]) or np.isnan(rsi[i]) or np.isnan(ema_200[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Range regime: Choppiness Index > 61.8
            if chop_4h[i] > 61.8:
                # Long: RSI oversold (<30) and price above EMA200 (bullish bias)
                if rsi[i] < 30 and price > ema_200[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: RSI overbought (>70) and price below EMA200 (bearish bias)
                elif rsi[i] > 70 and price < ema_200[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or chop regime ends
            if rsi[i] >= 50 or chop_4h[i] < 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or chop regime ends
            if rsi[i] <= 50 or chop_4h[i] < 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals