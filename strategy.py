#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index(14) for regime filter.
Enter long when KAMA up, RSI > 50, and choppy market (CHOP > 61.8) for mean reversion bounce.
Enter short when KAMA down, RSI < 50, and choppy market (CHOP > 61.8) for fade.
Use weekly trend filter (1w EMA50) to avoid counter-trend trades in strong trends.
ATR-based stoploss and discrete position sizing (0.25) to minimize fee drag.
Designed for low trade frequency (<25/year) to work in ranging/bear markets like 2025+.
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
    
    # === Indicators on primary (1d) timeframe ===
    # KAMA(10, 2, 30) - adaptive trend
    close_s = pd.Series(close)
    direction = np.abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = direction / volatility.replace(0, 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_up = close > kama
    kama_down = close < kama
    
    # RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_long = rsi > 50
    rsi_short = rsi < 50
    
    # Choppiness Index(14) - regime filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop_regime = chop > 61.8  # choppy/mean revert regime
    
    # === HTF: Weekly trend filter (1w EMA50) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    weekly_uptrend = close > ema_50_1w_aligned
    weekly_downtrend = close < ema_50_1w_aligned
    
    # === ATR(14) for stoploss ===
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of KAMA(10), RSI(14), CHOP(14), ATR(14), EMA50(1w)
    start_idx = max(10, 14, 14, 14, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        
        if position == 0:
            # Long: KAMA up, RSI > 50, choppy regime, AND weekly uptrend (avoid fighting strong weekly downtrend)
            long_signal = kama_up[i] and rsi_long[i] and chop_regime[i] and weekly_uptrend[i]
            
            # Short: KAMA down, RSI < 50, choppy regime, AND weekly downtrend (avoid fighting strong weekly uptrend)
            short_signal = kama_down[i] and rsi_short[i] and chop_regime[i] and weekly_downtrend[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA flips down OR RSI < 40 (momentum loss) OR ATR stoploss
            if (not kama_up[i]) or (rsi[i] < 40) or (close_val < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA flips up OR RSI > 60 (momentum loss) OR ATR stoploss
            if (not kama_down[i]) or (rsi[i] > 60) or (close_val > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0