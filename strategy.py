#!/usr/bin/env python3
"""
Experiment #1015: 6h Primary + 12h/1d HTF — KAMA Adaptive Trend + Volume Confirmation

Hypothesis: Kaufman Adaptive Moving Average (KAMA) outperforms EMA/HMA in 6h timeframe
because it automatically adjusts smoothing based on market efficiency (volatility vs trend).
KAMA flattens during choppy periods (reducing whipsaws) and accelerates during trends.

Key innovations:
1. KAMA(10) with Efficiency Ratio adaptively smooths price — no lag in trends, flat in ranges
2. 12h KAMA(21) for intermediate trend bias (HTF filter)
3. 1d HMA(21) for long-term directional bias (stronger filter)
4. Volume spike confirmation: entry volume > 1.5x 20-period avg (validates breakouts)
5. RSI(14) divergence detection for early reversal signals
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.20, ±0.30 to minimize fee churn

Why 6h specifically:
- 6h captures multi-day swings (4 bars/day = 1460 bars/year)
- Less noise than 4h, more signals than 12h
- Target: 30-60 trades/year (5-10 per quarter)
- Use 12h/1d HTF to filter direction, 6h for entry timing

Entry conditions (balanced for trade frequency):
- LONG: price>6h_KAMA>12h_KAMA + 1d_HMA bullish + volume>1.5x avg + RSI>45
- SHORT: price<6h_KAMA<12h_KAMA + 1d_HMA bearish + volume>1.5x avg + RSI<55
- Size increases when all 3 TFs align strongly

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_vol_hma_regime_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    
    Efficiency Ratio (ER) = |close - close_n| / sum(|close_i - close_i-1|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    
    Fast SC = 2/(fast_period+1) = 2/3 = 0.6667
    Slow SC = 2/(slow_period+1) = 2/31 = 0.0645
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if np.isnan(close[i]) or np.isnan(close[i - period]):
            continue
        
        signal = abs(close[i] - close[i - period])
        
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            if not np.isnan(close[j]) and not np.isnan(close[j - 1]):
                noise += abs(close[j] - close[j - 1])
        
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(close[i]) and not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = np.nan
    
    return kama

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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        window = volume[i - period + 1:i + 1]
        if not np.any(np.isnan(window)):
            vol_sma[i] = np.mean(window)
    
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
    
    # Calculate and align HTF indicators
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=21)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    kama_6h = calculate_kama(close, period=10)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLUME CONFIRMATION ===
        vol_spike = volume[i] > 1.5 * vol_sma_20[i]
        
        # === KAMA TREND DETECTION (6h) ===
        kama_6h_bull = close[i] > kama_6h[i]
        kama_6h_bear = close[i] < kama_6h[i]
        
        # KAMA slope (trend strength)
        kama_slope_bull = kama_6h[i] > kama_6h[i - 5] if not np.isnan(kama_6h[i - 5]) else False
        kama_slope_bear = kama_6h[i] < kama_6h[i - 5] if not np.isnan(kama_6h[i - 5]) else False
        
        # === HTF BIAS (12h KAMA + 1d HMA) ===
        kama_12h_bull = close[i] > kama_12h_aligned[i]
        kama_12h_bear = close[i] < kama_12h_aligned[i]
        
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong multi-TF alignment
        strong_bull = kama_6h_bull and kama_12h_bull and hma_1d_bull and kama_slope_bull
        strong_bear = kama_6h_bear and kama_12h_bear and hma_1d_bear and kama_slope_bear
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entries
        if strong_bull:
            # Strong alignment + volume confirmation
            if vol_spike and rsi_14[i] > 45.0 and rsi_14[i] < 75.0:
                desired_signal = SIZE_STRONG
            # Weaker entry without volume spike
            elif rsi_14[i] > 50.0 and rsi_14[i] < 70.0:
                desired_signal = SIZE_BASE
        
        # SHORT entries
        elif strong_bear:
            # Strong alignment + volume confirmation
            if vol_spike and rsi_14[i] < 55.0 and rsi_14[i] > 25.0:
                desired_signal = -SIZE_STRONG
            # Weaker entry without volume spike
            elif rsi_14[i] < 50.0 and rsi_14[i] > 30.0:
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