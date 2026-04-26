#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Chop_v1
Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, RSI(14) for momentum confirmation, and Choppiness Index(14) as regime filter to avoid ranging markets. Enter long when KAMA slope up, RSI > 50, and CHOP < 50 (trending regime). Enter short when KAMA slope down, RSI < 50, and CHOP < 50. Exit on opposite signal or when CHOP > 61.8 (strong ranging). Designed for low trade frequency (~10-25/year) to minimize fee drag and work in both bull and bear markets via adaptive trend filter and regime avoidance.
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
    
    # Get 1w data for HTF trend filter (optional, can use 1d EMA50 as proxy)
    # But per instruction: use 1d as primary, 1w as HTF
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for HTF trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # KAMA on 1d close (ER=10, fast=2, slow=30)
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]| over 10 periods)
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # sum of |diff| over window
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constant: SC = [ER * (fastest - slowest) + slowest]^2
    fastest = 2.0 / (2 + 1)
    slowest = 2.0 / (30 + 1)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # KAMA: kama[t] = kama[t-1] + SC * (close[t] - kama[t-1])
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # start after first ER can be calculated
    for i in range(10, n):
        if not np.isnan(sc[i-10]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14) on 1d close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element (diff reduces length by 1)
    rsi = np.concatenate([[np.nan], rsi])
    
    # Choppiness Index(14) on 1d: high CHOP = ranging, low CHOP = trending
    # CHOP = 100 * log10(sum(ATR over n) / (log(n) * (max(high) - min(low)))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first TR
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero or log of zero
    denominator = np.log(14) * (max_high - min_low)
    chop = np.where(denominator > 0, 
                    100 * np.log10(atr_sum / denominator) / np.log10(14), 
                    np.nan)
    
    # Align HTF indicators to 1d timeframe (prices is already 1d)
    # Since prices is 1d, alignment is trivial but we use the helper for consistency
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi)
    chop_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of KAMA needs 10, RSI needs 14, CHOP needs 14, EMA50 needs 50
    start_idx = max(10, 14, 14, 50)  # 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(kama_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        close_val = close[i]
        
        # KAMA slope: rising if close > kama, falling if close < kama
        kama_rising = close_val > kama_val
        kama_falling = close_val < kama_val
        
        # Trend filter: HTF 1w EMA50 direction
        uptrend_1w = close_val > ema_50_1w_val
        downtrend_1w = close_val < ema_50_1w_val
        
        # Regime filter: only trade when market is trending (CHOP < 50) or moderately choppy
        # Avoid strong ranging markets (CHOP > 61.8)
        regime_filter = chop_val < 50.0
        regime_exit = chop_val > 61.8  # exit if strong ranging
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, uptrend on 1w, favorable regime
            long_signal = kama_rising and \
                          (rsi_val > 50) and \
                          uptrend_1w and \
                          regime_filter
            
            # Short: KAMA falling, RSI < 50, downtrend on 1w, favorable regime
            short_signal = kama_falling and \
                           (rsi_val < 50) and \
                           downtrend_1w and \
                           regime_filter
            
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
            # Exit on opposite signal or strong ranging regime
            if (not kama_rising) or regime_exit or (rsi_val < 40):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit on opposite signal or strong ranging regime
            if (not kama_falling) or regime_exit or (rsi_val > 60):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_Chop_v1"
timeframe = "1d"
leverage = 1.0