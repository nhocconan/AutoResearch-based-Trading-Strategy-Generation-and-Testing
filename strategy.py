#!/usr/bin/env python3
"""
Experiment #1324: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + RSI(7) Pullback

Hypothesis: Recent failures show complex regime filters = 0 trades, simple HMA = whipsaw.
KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency ratio:
- Fast in trending markets (like EMA)
- Slow in choppy markets (like SMA)
This should reduce whipsaw while still generating trades.

Key changes from failed experiments:
1. KAMA instead of HMA (adapts to volatility automatically)
2. RSI(7) instead of RSI(14) — faster entry signals
3. Simpler regime logic (1d HMA bias only, no Choppiness/CRSI)
4. Volume filter to avoid low-liquidity entries
5. Loose entry bands to ensure ≥30 trades/year

Target: 25-50 trades/year, Sharpe > 0.612, DD < -30%
Timeframe: 4h
Size: 0.25-0.30 discrete levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi7_pullback_12h1d_bias_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average
    Adapts to market efficiency ratio (ER)
    ER = |net change| / sum of absolute changes
    High ER = trending (fast SC), Low ER = choppy (slow SC)
    """
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(period, n):
        net_change = abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
    
    # Smoothing Constant
    sc = np.full(n, np.nan)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
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

def calculate_sma(close, period=50):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_kama(df_1d['close'].values, period=21, fast=2, slow=30)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 12h KAMA for intermediate trend
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=14, fast=2, slow=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    # Calculate primary (4h) indicators
    kama_fast = calculate_kama(close, period=8, fast=2, slow=30)
    kama_slow = calculate_kama(close, period=21, fast=2, slow=30)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for entries
    atr = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, period=50)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(kama_12h_aligned[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # Volume filter (avoid low liquidity entries)
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === MACRO BIAS (1d KAMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (12h KAMA) ===
        inter_bull = close[i] > kama_12h_aligned[i]
        inter_bear = close[i] < kama_12h_aligned[i]
        
        # === LOCAL TREND (4h KAMA crossover) ===
        kama_bull = kama_fast[i] > kama_slow[i]
        kama_bear = kama_fast[i] < kama_slow[i]
        
        # === SMA50 FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Macro bull + volume ok + RSI pullback
        if macro_bull and volume_ok:
            # RSI pullback in uptrend (30-50 range) - LOOSE for more trades
            if 30.0 <= rsi[i] <= 50.0:
                # Require at least one trend confirmation
                if inter_bull or kama_bull or above_sma50:
                    desired_signal = BASE_SIZE
            # RSI breaking above 45 with momentum
            elif 45.0 < rsi[i] < 60.0 and kama_bull:
                desired_signal = BASE_SIZE
            # RSI oversold bounce (below 30) with macro support
            elif rsi[i] < 30.0 and inter_bull:
                desired_signal = BASE_SIZE
        
        # SHORT ENTRY: Macro bear + volume ok + RSI bounce
        elif macro_bear and volume_ok:
            # RSI bounce in downtrend (50-70 range) - LOOSE for more trades
            if 50.0 <= rsi[i] <= 70.0:
                # Require at least one trend confirmation
                if inter_bear or kama_bear or below_sma50:
                    desired_signal = -BASE_SIZE
            # RSI breaking below 55 with momentum
            elif 40.0 < rsi[i] < 55.0 and kama_bear:
                desired_signal = -BASE_SIZE
            # RSI overbought rejection (above 70) with macro resistance
            elif rsi[i] > 70.0 and inter_bear:
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
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