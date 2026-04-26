#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Chop_Filter_v2
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index(14) for regime filtering.
Enter long when KAMA trends up, RSI > 50, and market is trending (CHOP < 38.2).
Enter short when KAMA trends down, RSI < 50, and market is trending (CHOP < 38.2).
Exit on opposite signal or ATR-based stoploss.
Designed to capture strong trends while avoiding choppy markets where breakouts fail.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn.
Targets 30-100 trades over 4 years (7-25/year) on 1d timeframe to avoid fee drag.
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
    
    # Get 1w data for HTF trend filter (optional but adds robustness)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA(10,2,30) on daily prices
    # Efficiency ratio = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # For first 10 periods, volatility is cumulative sum of absolute changes
    er = np.zeros_like(close)
    er[10:] = change[10:] / np.maximum(volatility[10:], 1e-10)
    # Smoothing constants: fastest SC = 2/(2+1)=0.667, slowest SC = 2/(30+1)=0.0645
    sc = (er * (0.667 - 0.0645) + 0.0645) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed with close at period 10
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element (since diff reduces length by 1)
    rsi = np.concatenate([[np.nan], rsi])
    
    # Calculate Choppiness Index(14)
    # CHOP = 100 * log10(sum(ATR(1)) over 14 / (max(high)-min(low) over 14)) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1 = pd.Series(tr).ewm(alpha=1, adjust=False).mean().values  # ATR(1) = true range
    sum_tr_14 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    chop = np.zeros_like(close)
    chop[13:] = 100 * np.log10(sum_tr_14[13:] / np.maximum(range_14[13:], 1e-10)) / np.log10(14)
    
    # Calculate ATR(14) for stoploss
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Optional: 1w EMA34 for HTF trend filter (align to daily)
    if len(df_1w) >= 34:
        close_1w = df_1w['close'].values
        ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
        ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
        htf_trend_up = close > ema_34_1w_aligned
        htf_trend_down = close < ema_34_1w_aligned
    else:
        htf_trend_up = np.ones_like(close, dtype=bool)
        htf_trend_down = np.ones_like(close, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of KAMA seed(10), RSI(14), CHOP(14), ATR(14)
    start_idx = max(10, 14, 14, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(chop[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        kama_trend_up = close_val > kama[i]
        kama_trend_down = close_val < kama[i]
        rsi_mid = 50.0
        rsi_bullish = rsi[i] > rsi_mid
        rsi_bearish = rsi[i] < rsi_mid
        chop_trending = chop[i] < 38.2  # trending regime
        chop_choppy = chop[i] > 61.8    # choppy regime (avoid)
        vol_spike = False  # placeholder - volume not used in this version
        
        if position == 0:
            # Long: KAMA up, RSI > 50, trending regime, HTF trend up
            long_signal = (kama_trend_up and 
                          rsi_bullish and 
                          chop_trending and 
                          htf_trend_up[i])
            
            # Short: KAMA down, RSI < 50, trending regime, HTF trend down
            short_signal = (kama_trend_down and 
                           rsi_bearish and 
                           chop_trending and 
                           htf_trend_down[i])
            
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
            # Exit conditions: KAMA trends down OR RSI < 40 OR choppy OR stoploss
            if (kama_trend_down or 
                rsi[i] < 40 or 
                chop_choppy or 
                close_val < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions: KAMA trends up OR RSI > 60 OR choppy OR stoploss
            if (kama_trend_up or 
                rsi[i] > 60 or 
                chop_choppy or 
                close_val > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_Chop_Filter_v2"
timeframe = "1d"
leverage = 1.0