#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: On 1d timeframe, KAMA (adaptive trend) direction + RSI(14) extreme + Choppiness Index regime filter captures sustained moves while avoiding whipsaw in ranging markets. Uses discrete sizing (0.25) targeting 7-25 trades/year. Works in bull/bear by only taking KAMA-aligned signals. No stoploss needed - exit on signal reversal.
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === INDICATORS ===
    # KAMA(10, 2, 30) - Adaptive trend indicator
    close_s = pd.Series(close)
    change = np.abs(close_s.diff(10).values)
    volatility = np.abs(close_s.diff(1)).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Choppiness Index(14) - regime filter
    atr_14 = []
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros(n)
    for i in range(13, n):
        if atr_14[i] > 0 and max_high[i] > min_low[i]:
            sum_atr = np.nansum(atr_14[i-13:i+1])
            chop[i] = 100 * np.log10(sum_atr / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # === SIGNALS ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of KAMA(10), RSI(14), Chop(14)
    start_idx = max(10, 14, 14)
    
    for i in range(start_idx, n):
        kama_val = kama[i]
        close_val = close[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        
        # Skip if any data not ready
        if np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # KAMA direction: price > KAMA = uptrend, price < KAMA = downtrend
        is_uptrend = close_val > kama_val
        is_downtrend = close_val < kama_val
        
        # RSI extremes: <30 oversold, >70 overbought
        rsi_oversold = rsi_val < 30
        rsi_overbought = rsi_val > 70
        
        # Choppiness regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8
        
        # Entry conditions
        long_entry = is_uptrend and rsi_oversold and is_trending
        short_entry = is_downtrend and rsi_overbought and is_trending
        
        # Exit conditions: reverse signal or regime change to ranging
        long_exit = not is_uptrend or not rsi_oversold or is_ranging
        short_exit = not is_downtrend or not rsi_overbought or is_ranging
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0