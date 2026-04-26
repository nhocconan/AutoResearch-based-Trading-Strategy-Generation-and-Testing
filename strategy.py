#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: On 1d timeframe, enter long when KAMA(10) is rising AND RSI(14) > 50 AND Choppiness Index(14) < 38.2 (trending regime).
Enter short when KAMA(10) is falling AND RSI(14) < 50 AND Choppiness Index(14) < 38.2.
Exit when KAMA reverses direction or Choppiness Index > 61.8 (range regime).
Uses discrete sizing (0.0, ±0.25) to limit fee drag. Target: 7-25 trades/year.
Works in both bull and bear markets by adapting to trending regime via Chop filter.
"""

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
    
    # Get 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate KAMA(10) on 1d close
    close_s = pd.Series(close)
    # Efficiency Ratio (ER) over 10 periods
    change = abs(close_s - close_s.shift(10))
    volatility = abs(close_s.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    sc = sc.fillna(0.01)  # fallback when ER is NaN
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    kama = kama.astype(float)
    
    # KAMA direction: 1 if rising, -1 if falling, 0 if flat
    kama_dir = np.zeros_like(close)
    kama_dir[1:] = np.sign(np.diff(kama))
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Choppiness Index(14)
    atr = pd.Series(np.maximum(high - low, np.maximum(high - close_s.shift(1), low - close_s.shift(1)))).rolling(window=14, min_periods=14).mean()
    hh = pd.Series(high).rolling(window=14, min_periods=14).max()
    ll = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr.sum() / (hh - ll)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when undefined
    
    # 1w EMA50 for trend filter (aligned)
    close_1w = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA warmup (10), RSI warmup (14), ATR warmup (14), HH/LL warmup (14)
    start_idx = max(10, 14, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(kama_dir[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema_50_1w_aligned[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Entry conditions
        kama_rising = kama_dir[i] > 0
        kama_falling = kama_dir[i] < 0
        rsi_above_50 = rsi[i] > 50
        rsi_below_50 = rsi[i] < 50
        chop_trending = chop[i] < 38.2  # trending regime
        chop_ranging = chop[i] > 61.8   # range regime
        
        if position == 0:
            # Long: KAMA rising + RSI > 50 + trending regime
            long_signal = kama_rising and rsi_above_50 and chop_trending
            
            # Short: KAMA falling + RSI < 50 + trending regime
            short_signal = kama_falling and rsi_below_50 and chop_trending
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA reverses OR chop enters range regime
            if not kama_rising or chop_ranging:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA reverses OR chop enters range regime
            if not kama_falling or chop_ranging:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0