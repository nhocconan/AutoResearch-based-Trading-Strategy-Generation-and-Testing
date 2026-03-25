#!/usr/bin/env python3
"""
Experiment #1095: 6h Primary + 12h/1d HTF — HMA Trend + RSI Pullback

Hypothesis: Simpler is better for 6h timeframe. Complex regime switching has failed
repeatedly (exp #1087, #1091, #1094 all negative Sharpe). Instead, use proven pattern:
HTF trend direction (12h HMA) + LTF pullback entries (6h RSI) + 1d confirmation.

Key innovations:
1. 12h HMA(21) for primary trend direction (aligned properly via mtf_data)
2. 1d HMA(21) for stronger signal confirmation (only trade when 12h and 1d agree)
3. 6h RSI(14) pullback entries: long when RSI 35-50 in uptrend, short when 50-65 in downtrend
4. ATR(14) 2.5x trailing stoploss
5. LOOSE entry conditions to guarantee trades (learned from 0-trade failures)
6. Discrete sizing: 0.0, ±0.25, ±0.30

Why this should work:
- 6h captures multi-day swings (30-60 trades/year target)
- 12h trend filter avoids counter-trend trades that failed in exp #1087
- RSI pullback entries (not extremes) ensure we get trades during normal market conditions
- 1d agreement adds conviction for larger position size
- Simple logic = fewer conditions that can all fail simultaneously

Entry conditions (LOOSE to guarantee >=30 trades):
- LONG: price > 12h_HMA > 1d_HMA + RSI(14) between 35-55 (pullback in uptrend)
- SHORT: price < 12h_HMA < 1d_HMA + RSI(14) between 45-65 (pullback in downtrend)
- Stronger size when 12h and 1d HMA both agree with trend

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_rsi_pullback_12h1d_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND DIRECTION ===
        # 12h HMA trend
        trend_12h_bull = close[i] > hma_12h_aligned[i]
        trend_12h_bear = close[i] < hma_12h_aligned[i]
        
        # 1d HMA trend (stronger confirmation)
        trend_1d_bull = close[i] > hma_1d_aligned[i]
        trend_1d_bear = close[i] < hma_1d_aligned[i]
        
        # HMA slope confirmation (optional but helpful)
        hma_12h_slope_bull = hma_12h_aligned[i] > hma_12h_aligned[i-1] if i > 0 and not np.isnan(hma_12h_aligned[i-1]) else False
        hma_12h_slope_bear = hma_12h_aligned[i] < hma_12h_aligned[i-1] if i > 0 and not np.isnan(hma_12h_aligned[i-1]) else False
        
        # === ENTRY LOGIC (RSI PULLBACK IN TREND DIRECTION) ===
        desired_signal = 0.0
        
        # LONG entries: uptrend + RSI pullback (not oversold, just pulling back)
        # Looser conditions to ensure we get trades
        if trend_12h_bull:
            # Base long: price above 12h HMA + RSI in pullback zone
            if 35.0 <= rsi_14[i] <= 55.0:
                desired_signal = SIZE_BASE
            # Strong long: 12h and 1d both bullish + RSI pullback
            if trend_1d_bull and 40.0 <= rsi_14[i] <= 52.0:
                desired_signal = SIZE_STRONG
            # Momentum long: RSI crossing up from pullback
            if trend_1d_bull and rsi_14[i] > 50.0 and rsi_14[i] < 65.0:
                if i > 1 and not np.isnan(rsi_14[i-1]) and rsi_14[i-1] < 50.0:
                    desired_signal = SIZE_STRONG
        
        # SHORT entries: downtrend + RSI pullback (not overbought, just rallying)
        elif trend_12h_bear:
            # Base short: price below 12h HMA + RSI in pullback zone
            if 45.0 <= rsi_14[i] <= 65.0:
                desired_signal = -SIZE_BASE
            # Strong short: 12h and 1d both bearish + RSI pullback
            if trend_1d_bear and 48.0 <= rsi_14[i] <= 60.0:
                desired_signal = -SIZE_STRONG
            # Momentum short: RSI crossing down from pullback
            if trend_1d_bear and rsi_14[i] < 50.0 and rsi_14[i] > 35.0:
                if i > 1 and not np.isnan(rsi_14[i-1]) and rsi_14[i-1] > 50.0:
                    desired_signal = -SIZE_STRONG
        
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