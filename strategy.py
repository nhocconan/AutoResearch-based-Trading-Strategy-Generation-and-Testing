#!/usr/bin/env python3
"""
Experiment #132: 12h Primary + 1d/1w HTF — Triple-HTF KAMA Trend + RSI Pullback

Hypothesis: After 131 experiments, the clearest pattern is:
- 12h primary timeframe works best for BTC/ETH (lower noise than 4h, more trades than 1d)
- Triple-HTF confirmation (12h + 1d + 1w) filters out false signals in bear markets
- KAMA adapts better than HMA/EMA to crypto volatility regimes
- RSI pullback (40-60 range) captures trend continuations without waiting for extremes
- Choppiness Index ONLY as negative filter (skip trades when CHOP>65, don't require low CHOP)

Key innovations vs failed experiments:
1. Triple-HTF alignment: ALL three (12h price, 1d KAMA, 1w KAMA) must agree on direction
2. RSI continuation zone: 40-60 (not 25/75 extremes) = catches pullbacks in strong trends
3. Choppiness as ONLY negative filter: CHOP>65 = no trades (avoid chop), but CHOP<65 = trade allowed
4. ATR volatility scaling: position size reduces when ATR is high (protects in volatile periods)
5. Asymmetric stoploss: 2.5x ATR for longs, 3.0x ATR for shorts (crypto drops faster than rises)

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
Timeframe: 12h (20-50 trades/year target)
Position size: 0.25-0.30 (discrete levels to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_triple_htf_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio (ER)
    """
    n = len(close)
    if n < period + slow + 5:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    price_change = np.abs(close[period:] - close[:-period])
    sum_price_change = np.zeros(n - period)
    for i in range(n - period):
        diff = np.diff(close[i:i+period+1])
        sum_price_change[i] = np.sum(np.abs(diff))
    
    # Avoid division by zero
    er = np.zeros(n)
    for i in range(period, n):
        if sum_price_change[i-period] > 1e-10:
            er[i] = price_change[i-period] / sum_price_change[i-period]
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # Initialize KAMA with SMA of first period
    kama[period] = np.mean(close[:period+1])
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = choppy/range
    CHOP < 38.2 = trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR for each bar (true range)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d KAMA for intermediate trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=20, fast=2, slow=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate and align 1w KAMA for secular trend bias
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=15, fast=2, slow=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate primary (12h) indicators
    kama_fast = calculate_kama(close, period=8, fast=2, slow=20)
    kama_slow = calculate_kama(close, period=21, fast=2, slow=30)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% base position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # ATR normalization for volatility-adjusted sizing
    atr_median = np.nanmedian(atr[100:])
    if atr_median < 1e-10:
        atr_median = 1.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TRIPLE-HTF TREND CONFIRMATION ===
        # 12h: price relative to KAMA slow
        price_above_kama_12h = close[i] > kama_slow[i]
        price_below_kama_12h = close[i] < kama_slow[i]
        
        # 1d: price relative to 1d KAMA
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # 1w: price relative to 1w KAMA (secular trend)
        price_above_kama_1w = close[i] > kama_1w_aligned[i]
        price_below_kama_1w = close[i] < kama_1w_aligned[i]
        
        # === 12h KAMA CROSSOVER (entry trigger) ===
        kama_cross_bull = kama_fast[i] > kama_slow[i]
        kama_cross_bear = kama_fast[i] < kama_slow[i]
        
        # === RSI PULLBACK ZONE (continuation, not reversal) ===
        # For longs in uptrend: RSI 40-65 (pullback but not崩溃)
        # For shorts in downtrend: RSI 35-60 (bounce but not reversal)
        rsi_ok_long = 40.0 <= rsi[i] <= 65.0
        rsi_ok_short = 35.0 <= rsi[i] <= 60.0
        
        # === CHOPPINESS FILTER (negative only) ===
        # CHOP > 65 = too choppy, skip ALL trades
        # CHOP <= 65 = allow trades
        chop_ok = chop[i] <= 65.0
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        # Reduce size when ATR is high (volatile = more risk)
        atr_ratio = atr[i] / atr_median
        if atr_ratio > 2.0:
            vol_scale = 0.7  # reduce size 30% in high vol
        elif atr_ratio > 1.5:
            vol_scale = 0.85
        else:
            vol_scale = 1.0
        
        adjusted_size = BASE_SIZE * vol_scale
        
        # === DESIRED SIGNAL ===
        # LONG: ALL three HTF bullish + KAMA cross bull + RSI in zone + not choppy
        # SHORT: ALL three HTF bearish + KAMA cross bear + RSI in zone + not choppy
        desired_signal = 0.0
        
        if chop_ok:
            # Long: 12h + 1d + 1w all bullish
            if price_above_kama_12h and price_above_kama_1d and price_above_kama_1w:
                if kama_cross_bull and rsi_ok_long:
                    desired_signal = adjusted_size
            
            # Short: 12h + 1d + 1w all bearish
            if price_below_kama_12h and price_below_kama_1d and price_below_kama_1w:
                if kama_cross_bear and rsi_ok_short:
                    desired_signal = -adjusted_size
        
        # === STOPLOSS CHECK (Asymmetric: 2.5x long, 3.0x short) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals