#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Chop
Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(2) for mean-reversion entry timing, and Choppiness Index regime filter to avoid whipsaws.
KAMA adapts to market noise - fast in trends, slow in ranging markets. RSI(2) catches short-term
extremes within the trend. Chop filter ensures we only trade when market is trending (CHOP < 38.2)
or mean-reverting (CHOP > 61.8) appropriately. Designed for low trade frequency (7-25/year) with
discrete sizing (0.25) to minimize fee drag. Works in both bull and bear markets via regime adaptation.
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
    
    # Get 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # KAMA (Kaufman Adaptive Moving Average) - adapts to market efficiency
    # ER = Efficiency Ratio = |net change| / sum of absolute changes
    # Smoothest ER constant: 2/(fast+1) - 2/(slow+1)
    # We'll use fast=2, slow=30 for daily
    close_series = pd.Series(close)
    change = abs(close_series.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    net_change = abs(close_series.diff(10))
    er = net_change / volatility.replace(0, np.nan)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # smoothing constant
    kama = close_series.copy()
    kama.iloc[0] = close_series.iloc[0]
    for i in range(1, len(close_series)):
        if not np.isnan(sc.iloc[i]):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_series.iloc[i] - kama.iloc[i-1])
        else:
            kama.iloc[i] = kama.iloc[i-1]
    kama_values = kama.values
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama_values, additional_delay_bars=0)
    
    # RSI(2) for short-term mean reversion
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=2, min_periods=2).mean()
    avg_loss = loss.rolling(window=2, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Choppiness Index regime filter
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(N)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr3[0] = 0  # first period has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop_values = chop.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Discrete position size to minimize fee churn
    
    # Warmup: need KAMA (10), RSI(2), CHOP(14)
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_values[i]) or
            np.isnan(chop_values[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_values[i]
        chop_val = chop_values[i]
        
        if position == 0:
            # Regime-based entry logic
            if chop_val > 61.8:  # Ranging market - mean reversion
                # Long when RSI oversold and price above KAMA (bullish bias)
                if rsi_val < 25 and close_val > kama_val:
                    signals[i] = size
                    position = 1
                # Short when RSI overbought and price below KAMA (bearish bias)
                elif rsi_val > 75 and close_val < kama_val:
                    signals[i] = -size
                    position = -1
            else:  # Trending market (CHOP < 61.8) - follow momentum
                # Long when price above KAMA and RSI not extreme
                if close_val > kama_val and rsi_val > 40:
                    signals[i] = size
                    position = 1
                # Short when price below KAMA and RSI not extreme
                elif close_val < kama_val and rsi_val < 60:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA OR RSI overbought in ranging market
            if close_val < kama_val or (chop_val > 61.8 and rsi_val > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA OR RSI oversold in ranging market
            if close_val > kama_val or (chop_val > 61.8 and rsi_val < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_RSI_Chop"
timeframe = "1d"
leverage = 1.0