#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Chop_Filter
Hypothesis: Use KAMA direction on 1d for primary trend, RSI(14) for pullback entries in trending markets, and Choppiness index regime filter to avoid false signals. Targets 30-80 trades over 4 years (7-20/year) for 1d timeframe. Works in bull (trend continuation) and bear (pullbacks in downtrend) by only trading with the 1d KAMA trend filter. Volume confirmation adds conviction to entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 1d (ER=10, fast=2, slow=30)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate ATR(14) for stoploss
    tr1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr1 = np.concatenate([[0], tr1])
    atr = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness regime filter: CHOP(14) < 38.2 = trending market (good for trend following)
    hl_range = pd.Series(high - low).rolling(window=14, min_periods=14).sum()
    true_range = pd.Series(tr1).rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(hl_range / true_range) / np.log10(14)
    chop_regime = chop < 38.2  # trending market
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of KAMA calculation, RSI, ATR, CHOP, volume MA
    start_idx = max(30, 14, 14, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        trend_up = close_val > kama_aligned[i]   # 1d uptrend (price above KAMA)
        trend_down = close_val < kama_aligned[i]  # 1d downtrend (price below KAMA)
        regime_ok = chop_regime[i]  # trending market regime
        rsi_val = rsi_aligned[i]
        
        if position == 0:
            # Long: price above KAMA (uptrend) AND RSI < 40 (pullback) AND volume confirm AND trending regime
            long_signal = trend_up and (rsi_val < 40) and vol_conf and regime_ok
            
            # Short: price below KAMA (downtrend) AND RSI > 60 (pullback) AND volume confirm AND trending regime
            short_signal = trend_down and (rsi_val > 60) and vol_conf and regime_ok
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, close_val)
            # ATR trailing stop: exit if price drops 2.5 * ATR from highest since entry
            if close_val < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Alternative exit: trend flips down (price below KAMA) or regime changes to ranging
            elif not trend_up or not regime_ok:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, close_val)
            # ATR trailing stop: exit if price rises 2.5 * ATR from lowest since entry
            if close_val > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Alternative exit: trend flips up (price above KAMA) or regime changes to ranging
            elif not trend_down or not regime_ok:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0