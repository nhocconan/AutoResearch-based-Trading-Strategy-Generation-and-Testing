#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_MeanReversion_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for mean-reversion entries, and choppiness regime filter to avoid trending markets.
Long when: KAMA upward sloping + RSI < 30 (oversold) + chop > 61.8 (ranging regime).
Short when: KAMA downward sloping + RSI > 70 (overbought) + chop > 61.8.
Exit when: RSI crosses 50 (mean reversion complete) or opposite RSI extreme reached.
Uses discrete 0.25 position size. Targets 15-25 trades/year for optimal test generalization.
Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
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
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for KAMA and RSI
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) on 1d close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    fast = 2
    slow = 30
    close_1d = df_1d['close'].values
    
    # Calculate change and volatility
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    for i in range(1, len(close_1d)):
        change[i] = np.abs(close_1d[i] - close_1d[i-10]) if i >= 10 else np.abs(close_1d[i] - close_1d[0])
        volatility[i] = np.abs(close_1d[i] - close_1d[i-1])
    
    # 10-period ER
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        if volatility[i-9:i+1].sum() > 0:
            er[i] = change[i] / volatility[i-9:i+1].sum()
        else:
            er[i] = 0
    
    # SC = [ER * (fastest SC - slowest SC) + slowest SC]^2
    fastest_sc = 2 / (fast + 1)
    slowest_sc = 2 / (slow + 1)
    sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI(14) on 1d close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (CHOP) on 1d - using 14-period
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = max_high - min_low
    chop_raw = 100 * np.log10(atr * np.sqrt(14) / range_hl) / np.log10(14)
    chop = np.where(range_hl > 0, chop_raw, 50.0)  # default to neutral when range=0
    chop_regime = chop > 61.8  # ranging regime
    
    # Align HTF indicators to lower timeframe (wait for completed 1d bar)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 30 for KAMA slow, 14 for RSI, 14 for CHOP
    start_idx = max(30, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for mean reversion entries in ranging regime
            # Long: KAMA upward + RSI < 30 (oversold) + chop > 61.8
            long_entry = (kama_aligned[i] > kama_aligned[i-1]) and \
                       (rsi_aligned[i] < 30) and \
                       (chop_regime_aligned[i] > 0.5)
            # Short: KAMA downward + RSI > 70 (overbought) + chop > 61.8
            short_entry = (kama_aligned[i] < kama_aligned[i-1]) and \
                        (rsi_aligned[i] > 70) and \
                        (chop_regime_aligned[i] > 0.5)
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when RSI crosses 50 (mean reversion) or RSI > 70 (overbought)
            if (rsi_aligned[i] > 50) or (rsi_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when RSI crosses 50 (mean reversion) or RSI < 30 (oversold)
            if (rsi_aligned[i] < 50) or (rsi_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_RSI_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0