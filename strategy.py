#!/usr/bin/env python3
"""
Experiment #1563: 6h Primary + 1d/1w HTF — KAMA Adaptive Trend with RSI Pullback

Hypothesis: 6h timeframe captures multi-day crypto swings better than 4h or 12h.
KAMA (Kaufman Adaptive Moving Average) adapts smoothing based on market efficiency,
performing better than fixed EMA/HMA in crypto's varying volatility regimes.

Key components:
1. 1w HMA(21) for major secular trend bias (very slow, avoids counter-trend)
2. 1d HMA(21) for intermediate trend confirmation
3. 6h KAMA(21) for adaptive entry timing (responds to volatility changes)
4. RSI(14) pullback entries within trend (buy dips in uptrend, sell rallies in downtrend)
5. ROC(10) momentum confirmation (ensures moves have strength)
6. ATR(14) trailing stoploss (2.5x ATR)
7. Volatility regime filter (ATR ratio 7/30) to adjust position size

Why this should work on 6h:
- KAMA reduces whipsaws in chop while catching trends quickly
- Dual HTF filter (1w + 1d) prevents major counter-trend positions
- RSI pullback entries = better risk/reward than breakout chasing
- LOOSE thresholds (RSI <50/>50, ROC >1%) guarantee sufficient trades
- 6h TF naturally targets 30-60 trades/year (fee-efficient)

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: 1w_HMA bullish + 1d_HMA bullish + KAMA rising + RSI<50 + ROC>1%
- SHORT: 1w_HMA bearish + 1d_HMA bearish + KAMA falling + RSI>50 + ROC<-1%
- Size increased in high vol regime (ATR ratio >1.5)

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_rsi_pullback_1w1d_v1"
timeframe = "6h"
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

def calculate_kama(close, er_period=10, fast_sc=2/11, slow_sc=2/31):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    if n < er_period + 1:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(close[i]) or np.isnan(close[i - er_period]):
            continue
        
        # Efficiency Ratio: net change / sum of absolute changes
        net_change = abs(close[i] - close[i - er_period])
        sum_changes = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        
        if sum_changes > 1e-10:
            er = net_change / sum_changes
        else:
            er = 0.0
        
        # Smoothing constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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
        if close[i - period] > 1e-10:
            roc[i] = 100.0 * (close[i] - close[i - period]) / close[i - period]
    
    return roc

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    kama_21 = calculate_kama(close, er_period=10)
    rsi_14 = calculate_rsi(close, period=14)
    roc_10 = calculate_roc(close, period=10)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    # ATR ratio for volatility regime
    atr_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(30, n):
        if atr_30[i] > 1e-10:
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_HIGH_VOL = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 60
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_21[i]) or np.isnan(roc_10[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY REGIME ===
        vol_ratio = atr_ratio[i]
        high_vol = vol_ratio > 1.5
        current_size = SIZE_HIGH_VOL if high_vol else SIZE_BASE
        
        # === TREND BIAS (HTF) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === KAMA SLOPE (adaptive trend) ===
        kama_slope = kama_21[i] - kama_21[i - 3] if i >= 3 else 0.0
        kama_rising = kama_slope > 0.0
        kama_falling = kama_slope < 0.0
        
        # === RSI ===
        rsi = rsi_14[i]
        
        # === ROC MOMENTUM ===
        roc = roc_10[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1w bullish + 1d bullish + KAMA rising + RSI pullback + ROC positive
        if price_above_1w and price_above_1d and kama_rising:
            if rsi < 50 and roc > 1.0:
                desired_signal = current_size
            elif rsi < 45 and roc > 0.5:
                desired_signal = current_size
        
        # SHORT: 1w bearish + 1d bearish + KAMA falling + RSI rally + ROC negative
        elif price_below_1w and price_below_1d and kama_falling:
            if rsi > 50 and roc < -1.0:
                desired_signal = -current_size
            elif rsi > 55 and roc < -0.5:
                desired_signal = -current_size
        
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
        if desired_signal >= SIZE_HIGH_VOL * 0.9:
            final_signal = SIZE_HIGH_VOL
        elif desired_signal <= -SIZE_HIGH_VOL * 0.9:
            final_signal = -SIZE_HIGH_VOL
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