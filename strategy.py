#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter
Hypothesis: On daily timeframe, Kaufman Adaptive Moving Average (KAMA) identifies the trend direction, RSI(14) provides momentum confirmation, and Choppiness Index(14) filters for trending regimes (CHOP < 38.2) to avoid whipsaws in ranging markets. This combination works in both bull and bear markets by only taking trades aligned with the adaptive trend during trending conditions, minimizing false signals. Discrete sizing (±0.25) and no additional trailing stop (rely on signal reversal) keeps trades low (~15-25/year) to reduce fee drag, with BTC/ETH edge from trend adherence during strong moves.
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
    
    # Load 1w data ONCE before loop for trend context (optional filter)
    df_1w = get_htf_data(prices, '1w')
    
    # Primary indicators on 1d timeframe (loaded via get_htf_data for alignment)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for KAMA, RSI, CHOP
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === KAMA (Kaufman Adaptive Moving Average) ===
    # ER = Efficiency Ratio, smoothed with fast/slow SC
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0) if False else None  # placeholder
    # Correct calculation:
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    # Volatility = sum of absolute changes over lookback period
    volatility = pd.Series(close_1d).rolling(window=10, min_periods=1).apply(lambda x: np.sum(np.abs(np.diff(x, prepend=x[0]))), raw=True)
    # Avoid division by zero
    volatility = volatility.values
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 1d timeframe (already 1d, but using align for consistency with HTF pattern)
    # Since we're on 1d timeframe, direct use is fine, but we'll align anyway for pattern
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === RSI(14) ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === Choppiness Index (CHOP) ===
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    chop = np.divide(atr_sum, (hh - ll), out=np.full_like(atr_sum, 50.0), where=(hh - ll)!=0) * 100
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === Signals ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # 25% position size
    
    # Warmup: max of KAMA initialization (1), RSI (14), CHOP (14)
    start_idx = max(1, 14, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val)):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend direction: price above/below KAMA
        trend_bullish = close_val > kama_val
        trend_bearish = close_val < kama_val
        
        # RSI momentum: not overbought/oversold extreme
        rsi_not_extreme = (rsi_val > 20) & (rsi_val < 80)
        
        # Regime filter: trending market (CHOP < 38.2)
        trending_regime = chop_val < 38.2
        
        # Entry conditions
        long_entry = trend_bullish and rsi_not_extreme and trending_regime
        short_entry = trend_bearish and rsi_not_extreme and trending_regime
        
        # Exit on signal reversal (opposite entry) or regime change to ranging
        long_exit = not trend_bullish or not rsi_not_extreme or not trending_regime
        short_exit = not trend_bearish or not rsi_not_extreme or not trending_regime
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0