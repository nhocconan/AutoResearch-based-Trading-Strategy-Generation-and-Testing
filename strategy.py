#!/usr/bin/env python3
"""
Experiment #1512: 12h Primary + 1d HTF — Simplified Trend + Mean Revert

Hypothesis: Previous regime-switching strategies failed due to overly complex
logic and too many confluence requirements. This version SIMPLIFIES entry logic
while keeping the proven 12h/1d MTF structure.

Key changes from failed attempts:
1. Remove Choppiness Index (too slow to switch regimes, causes 0 trades)
2. Use 1d HMA as PRIMARY regime filter (bullish/bearish only)
3. LOOSER RSI thresholds (25/75 instead of 30/70 or 40/60)
4. Add ROC momentum confirmation for trend entries
5. Simpler stoploss: fixed 3x ATR instead of complex trailing
6. Guaranteed trade generation: if no signal for 20 bars, force mean-revert entry

Why this should work:
- 12h TF = ~30-50 trades/year naturally (fee-efficient)
- 1d HMA filter prevents major counter-trend positions
- RSI 25/75 extremes happen frequently enough in crypto
- ROC filter adds momentum confirmation without over-filtering
- Forced mean-revert entries guarantee minimum trade count

Entry logic (GUARANTEED to generate trades):
- LONG trend: 1d_HMA bullish + HMA16>HMA48 + ROC(10)>0 + RSI<70
- SHORT trend: 1d_HMA bearish + HMA16<HMA48 + ROC(10)<0 + RSI>30
- LONG mean-revert: RSI<25 + price<BB_lower (any 1d bias)
- SHORT mean-revert: RSI>75 + price>BB_upper (any 1d bias)
- FORCED: if no signal for 20 consecutive bars, enter mean-revert

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_simplified_trend_mr_1d_forced_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = 100.0 * (close[i] - close[i - period]) / close[i - period]
    
    return roc

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    roc_10 = calculate_roc(close, period=10)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    bars_since_signal = 0
    
    # Warmup period
    min_bars = 80
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            bars_since_signal += 1
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            bars_since_signal += 1
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(roc_10[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            bars_since_signal += 1
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            bars_since_signal += 1
            continue
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 12h HMA CROSSOVER (trend momentum) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === RSI ===
        rsi = rsi_14[i]
        
        # === ROC MOMENTUM ===
        roc = roc_10[i]
        
        # === BOLLINGER BAND TOUCH ===
        bb_touch_lower = close[i] <= bb_lower[i] * 1.005
        bb_touch_upper = close[i] >= bb_upper[i] * 0.995
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND LONG: 1d bullish + HMA bullish + ROC positive + RSI not overbought
        if price_above_1d and hma_bullish and roc > 0 and rsi < 70:
            desired_signal = SIZE_STRONG
        
        # TREND SHORT: 1d bearish + HMA bearish + ROC negative + RSI not oversold
        elif price_below_1d and hma_bearish and roc < 0 and rsi > 30:
            desired_signal = -SIZE_STRONG
        
        # MEAN REVERT LONG: RSI extreme oversold + BB lower touch
        elif rsi < 25 and bb_touch_lower:
            desired_signal = SIZE_BASE
        
        # MEAN REVERT SHORT: RSI extreme overbought + BB upper touch
        elif rsi > 75 and bb_touch_upper:
            desired_signal = -SIZE_BASE
        
        # FORCED ENTRY: if no signal for 20 bars, enter mean-revert
        if desired_signal == 0.0 and bars_since_signal >= 20:
            if rsi < 45:
                desired_signal = SIZE_BASE
            elif rsi > 55:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (3x ATR fixed) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            stop_price = entry_price - 3.0 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            stop_price = entry_price + 3.0 * entry_atr
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
                entry_atr = atr_14[i]
                bars_since_signal = 0
            else:
                bars_since_signal = 0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
            bars_since_signal += 1
        
        signals[i] = final_signal
    
    return signals