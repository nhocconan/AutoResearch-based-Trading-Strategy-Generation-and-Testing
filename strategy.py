#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_MeanReversion_ChopFilter
Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI for mean-reversion entries and Choppiness Index for regime filtering.
Long when: KAMA trend up, RSI < 30 (oversold), and market is choppy (CHOP > 61.8).
Short when: KAMA trend down, RSI > 70 (overbought), and market is choppy (CHOP > 61.8).
This strategy aims to catch mean-reversion moves within choppy regimes while avoiding
strong trends where mean reversion fails. Designed for low trade frequency (7-25/year)
to minimize fee drag on 1d timeframe.
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
    
    # Calculate 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    # Calculate 1w EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d close
    # KAMA parameters: ER period=10, Fast SC=2/(2+1)=0.6667, Slow SC=2/(30+1)=0.0645
    close_series = pd.Series(close)
    change = abs(close_series - close_series.shift(10))
    volatility = close_series.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, 1e-10)  # Avoid division by zero
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate Choppiness Index (CHOP) on 1d
    # CHOP = 100 * log10(sum(ATR(1), n) / (max(high, n) - min(low, n))) / log10(n)
    # Using period=14
    tr1 = high - low
    tr2 = abs(high - np.roll(close, 1))
    tr3 = abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop_range = max_high - min_low
    chop_range_safe = np.where(chop_range == 0, 1e-10, chop_range)
    chop = 100 * np.log10(chop_sum / chop_range_safe) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # 25% position size
    
    # Warmup: need enough for KAMA (10), EMA20_1w (20), RSI (14), CHOP (14)
    start_idx = max(20, 14, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(rsi_values[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama[i]
        ema_trend = ema_20_1w_aligned[i]
        rsi_val = rsi_values[i]
        chop_val = chop[i]
        
        # Regime filter: only trade in choppy markets (CHOP > 61.8)
        in_choppy_regime = chop_val > 61.8
        
        if position == 0:
            # Flat - look for entry
            # Long: KAMA trend up (price > KAMA), RSI oversold (<30), choppy regime
            # Short: KAMA trend down (price < KAMA), RSI overbought (>70), choppy regime
            long_condition = (close_val > kama_val) and (rsi_val < 30) and in_choppy_regime
            short_condition = (close_val < kama_val) and (rsi_val > 70) and in_choppy_regime
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Long - exit when: price crosses below KAMA OR RSI > 50 (mean reversion complete)
            if close_val < kama_val or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when: price crosses above KAMA OR RSI < 50 (mean reversion complete)
            if close_val > kama_val or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_RSI_MeanReversion_ChopFilter"
timeframe = "1d"
leverage = 1.0