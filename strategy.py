#!/usr/bin/env python3
"""
Experiment #720: 6h Primary + 1d/1w HTF — Keltner-BB Squeeze + RSI Momentum

Hypothesis: 6h timeframe captures multi-day swings with reduced noise vs 4h.
Keltner-BB squeeze detects volatility compression before breakouts.
Unlike Donchian breakouts (failed on 6h), squeeze patterns catch consolidation
before major moves - works in both bull and bear markets.

Key innovations:
1. Bollinger Band Width percentile (20-day) for squeeze detection
2. Keltner Channel (ATR 1.5x) for volatility envelope
3. BB inside Keltner = squeeze (low vol compression)
4. BB expands outside Keltner + RSI momentum = breakout signal
5. 1d HMA(21) for intermediate trend filter
6. 1w HMA(21) for major bias confirmation
7. ATR(14) 2.5x trailing stop for risk management
8. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (LOOSE to ensure trades on all symbols):
- LONG: squeeze detected + BB expansion up + RSI>50 + 1d HMA bull + 1w HMA bull
- SHORT: squeeze detected + BB expansion down + RSI<50 + 1d HMA bear + 1w HMA bear
- Relaxed: only need 1d HMA alignment (not both 1d+1w) for more trades

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_keltner_bb_squeeze_rsi_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands - volatility envelope"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = upper - lower
    
    return upper, lower, width

def calculate_keltner_channel(high, low, close, period=20, atr_mult=1.5):
    """Keltner Channel - ATR-based volatility envelope"""
    n = len(close)
    if n < period + 14:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, period=14)
    
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    
    return upper, lower, atr

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_bb_width_percentile(bb_width, lookback=60):
    """Percentile rank of BB Width - detects volatility compression"""
    n = len(bb_width)
    percentile = np.zeros(n)
    percentile[:] = np.nan
    
    for i in range(lookback, n):
        if np.isnan(bb_width[i]):
            continue
        window = bb_width[i-lookback:i]
        valid = window[~np.isnan(window)]
        if len(valid) < lookback // 2:
            continue
        rank = np.sum(valid < bb_width[i])
        percentile[i] = 100.0 * rank / len(valid)
    
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    keltner_upper, keltner_lower, atr = calculate_keltner_channel(high, low, close, period=20, atr_mult=1.5)
    rsi = calculate_rsi(close, period=14)
    bb_width_pct = calculate_bb_width_percentile(bb_width, lookback=60)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(keltner_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = False
        htf_1w_bear = False
        if not np.isnan(hma_1w_aligned[i]):
            htf_1w_bull = close[i] > hma_1w_aligned[i]
            htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === SQUEEZE DETECTION (BB inside Keltner) ===
        squeeze_active = (bb_upper[i] <= keltner_upper[i]) and (bb_lower[i] >= keltner_lower[i])
        
        # === BB WIDTH PERCENTILE (volatility compression) ===
        low_vol_compression = False
        if not np.isnan(bb_width_pct[i]):
            low_vol_compression = bb_width_pct[i] < 30.0  # Bottom 30% = compression
        
        # === BB EXPANSION (breakout from squeeze) ===
        bb_expansion_up = False
        bb_expansion_down = False
        if i > 0 and not np.isnan(bb_upper[i-1]) and not np.isnan(keltner_upper[i-1]):
            # BB upper breaks above Keltner upper = bullish expansion
            bb_expansion_up = bb_upper[i] > keltner_upper[i] and close[i] > bb_upper[i-1]
            # BB lower breaks below Keltner lower = bearish expansion
            bb_expansion_down = bb_lower[i] < keltner_lower[i] and close[i] < bb_lower[i-1]
        
        # === RSI MOMENTUM ===
        rsi_bull = rsi[i] > 50.0
        rsi_bear = rsi[i] < 50.0
        rsi_strong_bull = rsi[i] > 55.0
        rsi_strong_bear = rsi[i] < 45.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: squeeze + expansion up + RSI bull + 1d HMA bull
        if (squeeze_active or low_vol_compression) and bb_expansion_up:
            if rsi_strong_bull and htf_1d_bull:
                desired_signal = SIZE_STRONG
            elif rsi_bull and htf_1d_bull:
                desired_signal = SIZE_BASE
        
        # LONG: 1w HMA bull + RSI oversold bounce (mean reversion in bull trend)
        elif htf_1w_bull and rsi[i] < 35.0:
            desired_signal = SIZE_BASE
        
        # SHORT: squeeze + expansion down + RSI bear + 1d HMA bear
        elif (squeeze_active or low_vol_compression) and bb_expansion_down:
            if rsi_strong_bear and htf_1d_bear:
                desired_signal = -SIZE_STRONG
            elif rsi_bear and htf_1d_bear:
                desired_signal = -SIZE_BASE
        
        # SHORT: 1w HMA bear + RSI overbought rejection (mean reversion in bear trend)
        elif htf_1w_bear and rsi[i] > 65.0:
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
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
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals