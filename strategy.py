#!/usr/bin/env python3
"""
Experiment #131: 4h Primary + 1d HTF — Asymmetric Regime Trend Following

Hypothesis: After 130 experiments, clear patterns emerge:
1. Symmetric long/short strategies fail in 2022 crash and 2025 bear market
2. ASYMMETRIC entries work better: only long in bull regime, only short in bear regime
3. KAMA adapts to volatility better than HMA/EMA (proven in baseline)
4. RSI pullback entries (not breakouts) have higher win rate in trending markets
5. Volume confirmation reduces false signals significantly

This strategy uses asymmetric regime logic:
1. 1d KAMA = regime detector (price above = bull, below = bear)
2. 4h KAMA(21) = dynamic support/resistance for pullback entries
3. RSI(14) pullback: long when RSI 35-55 in bull, short when RSI 45-65 in bear
4. Volume filter: entry volume > 0.8 * 20-bar avg volume
5. ATR trailing stop (2.5x) for risk management

Key differences from failed experiments:
- NO symmetric entries (no long in bear, no short in bull)
- RSI pullback range (not extreme oversold/overbought)
- Volume confirmation required
- Simpler than CRSI/Choppiness regimes that failed (#120-127, #130)

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
Timeframe: 4h (20-50 trades/year target)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_asymmetric_pullback_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio (ER)
    """
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    price_change = np.abs(close[period:] - close[:-period])
    sum_price_change = np.zeros(n - period)
    for i in range(n - period):
        diff_vals = np.abs(np.diff(close[i:i+period+1]))
        sum_price_change[i] = np.sum(diff_vals) if len(diff_vals) > 0 else 1e-10
    
    # Avoid division by zero
    er = np.zeros(n)
    for i in range(period, n):
        idx = i - period
        if sum_price_change[idx] > 1e-10:
            er[i] = price_change[idx] / sum_price_change[idx]
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
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(values, period):
    """Simple Moving Average"""
    n = len(values)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.zeros(n)
    sma[:] = np.nan
    for i in range(period - 1, n):
        sma[i] = np.mean(values[i-period+1:i+1])
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA for regime detection
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21, fast=2, slow=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=21, fast=2, slow=30)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_sma(volume, 20)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative for 4h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_4h[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (1d KAMA) ===
        # Bull regime: price above daily KAMA
        # Bear regime: price below daily KAMA
        bull_regime = close[i] > kama_1d_aligned[i]
        bear_regime = close[i] < kama_1d_aligned[i]
        
        # === 4h PULLBACK DETECTION ===
        # In bull regime: look for pullback to KAMA support
        # In bear regime: look for rally to KAMA resistance
        price_to_kama_ratio = close[i] / kama_4h[i] if kama_4h[i] > 1e-10 else 1.0
        
        # Pullback in bull: price slightly below or at KAMA (0.98-1.02)
        pullback_long = 0.97 <= price_to_kama_ratio <= 1.03
        
        # Rally in bear: price slightly above or at KAMA (0.97-1.03)
        rally_short = 0.97 <= price_to_kama_ratio <= 1.03
        
        # === RSI PULLBACK FILTER ===
        # Bull regime: RSI 35-60 (pullback but not crash)
        # Bear regime: RSI 40-65 (rally but not breakout)
        rsi_long_ok = 35.0 <= rsi[i] <= 60.0
        rsi_short_ok = 40.0 <= rsi[i] <= 65.0
        
        # === VOLUME CONFIRMATION ===
        # Entry volume should be at least 80% of 20-bar average
        vol_ok = volume[i] >= 0.8 * vol_sma[i]
        
        # === ASYMMETRIC ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Bull regime + pullback to KAMA + RSI in range + volume ok
        if bull_regime and pullback_long and rsi_long_ok and vol_ok:
            desired_signal = SIZE
        
        # SHORT: Bear regime + rally to KAMA + RSI in range + volume ok
        elif bear_regime and rally_short and rsi_short_ok and vol_ok:
            desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
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