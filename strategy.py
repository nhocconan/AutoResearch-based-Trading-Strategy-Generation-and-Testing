#!/usr/bin/env python3
"""
Experiment #251: 6h Primary + 1d/1w HTF — Volatility Regime + KAMA Adaptive Trend + RSI Mean Reversion

Hypothesis: 6h timeframe sits between 4h (too noisy) and 12h (too slow). Key insight from failures:
volatility-based regime detection works better than CHOP for crypto's explosive moves.

REGIME DETECTION (Volatility-Based):
- ATR(7)/ATR(28) ratio > 1.8 = HIGH VOL → Mean reversion (fade RSI extremes)
- ATR(7)/ATR(28) ratio < 1.2 = LOW VOL → Trend following (KAMA breakout)
- 1.2-1.8 = transition (use previous regime memory)

ENTRY LOGIC:
- HIGH VOL REGIME: RSI(7) < 25 + price > 1d HMA → Long (oversold bounce in uptrend)
                RSI(7) > 75 + price < 1d HMA → Short (overbought fade in downtrend)
- LOW VOL REGIME: Price breaks KAMA(21) + 1w HMA confirms → Trend entry
                KAMA slope confirms direction

HTF FILTERS:
- 1d HMA(34): Intermediate trend (only trade with 1d direction in low vol)
- 1w HMA(21): Major trend bias (required for strong signals)

KAMA (Kaufman Adaptive Moving Average):
- ER (Efficiency Ratio) = |close - close[n]| / sum(|close[i] - close[i-1]|)
- Fast SC = 2/(2+1), Slow SC = 2/(20+1)
- KAMA adapts smoothing based on market noise

Position sizing: 0.25 base, 0.30 strong (with 1w confirmation)
Stoploss: 2.0x ATR trailing (tighter than 12h due to more trades)

Target: Beat Sharpe=0.399, DD>-40%, trades>=30 train, trades>=3 test on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vol_regime_kama_rsi_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        noise = 0.0
        for j in range(i - er_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        
        if noise > 1e-10:
            er[i] = price_change / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Initialize KAMA with SMA of first er_period bars
    kama[er_period] = np.mean(close[:er_period + 1])
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
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

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=34)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_28 = calculate_atr(high, low, close, period=28)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate volatility ratio for regime detection
    vol_ratio = np.zeros(n)
    vol_ratio[:] = np.nan
    for i in range(28, n):
        if atr_28[i] > 1e-10 and not np.isnan(atr_7[i]):
            vol_ratio[i] = atr_7[i] / atr_28[i]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=low_vol_trend, 2=high_vol_mr
    prev_kama_slope = 0
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(kama[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === VOLATILITY REGIME DETECTION with HYSTERESIS ===
        high_vol_threshold = 1.8
        low_vol_threshold = 1.2
        
        if vol_ratio[i] > high_vol_threshold:
            current_regime = 2  # high vol → mean reversion
        elif vol_ratio[i] < low_vol_threshold:
            current_regime = 1  # low vol → trend following
        else:
            current_regime = prev_regime  # use memory
        
        prev_regime = current_regime
        
        # === KAMA SLOPE ===
        kama_slope = 0.0
        if i >= 3 and not np.isnan(kama[i-3]):
            kama_slope = (kama[i] - kama[i-3]) / kama[i-3] if kama[i-3] > 1e-10 else 0.0
        prev_kama_slope = kama_slope
        
        # === HTF BIAS ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # 1w for major trend
        htf_1w_valid = not np.isnan(hma_1w_aligned[i])
        htf_1w_bull = htf_1w_valid and close[i] > hma_1w_aligned[i]
        htf_1w_bear = htf_1w_valid and close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = False
        rsi_overbought = False
        if not np.isnan(rsi_7[i]):
            rsi_oversold = rsi_7[i] < 25.0
            rsi_overbought = rsi_7[i] > 75.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: LOW VOL (trend following with KAMA)
        if current_regime == 1:
            # Long: KAMA bullish + KAMA slope up + 1d HMA bull
            if kama_bull and kama_slope > 0.001 and htf_1d_bull:
                if htf_1w_bull:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # Short: KAMA bearish + KAMA slope down + 1d HMA bear
            elif kama_bear and kama_slope < -0.001 and htf_1d_bear:
                if htf_1w_bear:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # REGIME 2: HIGH VOL (mean reversion with RSI)
        elif current_regime == 2:
            # Long: RSI oversold + above SMA200 + 1d HMA bull (fade the dip in uptrend)
            if rsi_oversold and above_sma200 and htf_1d_bull:
                if htf_1w_bull:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # Short: RSI overbought + below SMA200 + 1d HMA bear (fade the rip in downtrend)
            elif rsi_overbought and below_sma200 and htf_1d_bear:
                if htf_1w_bear:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
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