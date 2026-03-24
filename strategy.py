#!/usr/bin/env python3
"""
Experiment #094: 4h Primary + 1d HTF — HMA Trend + Volume + ATR Vol Filter

Hypothesis: After analyzing 70+ failed experiments, the pattern is clear:
- Overly complex filters (Choppiness, Connors RSI, Fisher) = 0 trades
- Simple HMA trend works but needs volume confirmation to avoid false breakouts
- ATR volatility filter prevents trading during extreme conditions (panic/euphoria)
- 4h timeframe with 1d HTF bias should generate 20-50 trades/year with good Sharpe

This strategy combines:
1. 1d HMA(50) = major trend bias (HTF filter)
2. 4h HMA(16/48) crossover = entry trigger
3. Volume ratio filter (vol > 0.8 * avg_vol) = confirms breakout validity
4. ATR ratio filter (0.5 < ATR(7)/ATR(30) < 2.5) = avoids extreme vol conditions
5. RSI loose filter (>40 long, <60 short) = avoids extreme overbought/oversold
6. Trailing ATR stoploss (2.5x) = risk management
7. Position size: 0.28 (28% of capital, conservative for 4h)

Key design choices:
- Timeframe: 4h (proven to work, target 20-50 trades/year)
- HTF: 1d for trend bias (matches Binance actual 1d bars via mtf_data)
- Volume filter: avoids low-volume false breakouts
- ATR ratio: 0.5-2.5 range avoids panic/euphoria extremes
- RSI: 40/60 thresholds (looser than 30/70, ensures trades generate)
- Stoploss: 2.5x ATR trailing (tighter than 3x for better risk control)

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_vol_atr_regime_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_series = pd.Series(close)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = close_series.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma_full = close_series.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    
    return hma.values

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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio: current volume / rolling average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_fast = calculate_hma(close, period=16)
    hma_slow = calculate_hma(close, period=48)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    atr_fast = calculate_atr(high, low, close, period=7)
    atr_slow = calculate_atr(high, low, close, period=30)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 4h)
    
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
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_ratio[i]) or np.isnan(atr_fast[i]) or np.isnan(atr_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if atr_slow[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND (HMA crossover) ===
        hma_cross_bull = hma_fast[i] > hma_slow[i]
        hma_cross_bear = hma_fast[i] < hma_slow[i]
        
        # === VOLUME FILTER ===
        # Volume must be at least 80% of average to confirm breakout
        vol_ok = vol_ratio[i] >= 0.8
        
        # === ATR VOLATILITY FILTER ===
        # ATR ratio between 0.5 and 2.5 = normal vol conditions
        # Avoid trading during extreme panic (ratio > 2.5) or dead markets (ratio < 0.5)
        atr_ratio = atr_fast[i] / atr_slow[i]
        vol_regime_ok = 0.5 <= atr_ratio <= 2.5
        
        # === RSI FILTER (LOOSE - ensure trades generate) ===
        rsi_ok_long = rsi[i] > 40.0
        rsi_ok_short = rsi[i] < 60.0
        
        # === DESIRED SIGNAL ===
        # LONG: 1d bull + 4h HMA cross bull + volume ok + vol regime ok + RSI > 40
        # SHORT: 1d bear + 4h HMA cross bear + volume ok + vol regime ok + RSI < 60
        desired_signal = 0.0
        
        if htf_bull and hma_cross_bull and vol_ok and vol_regime_ok and rsi_ok_long:
            desired_signal = SIZE
        elif htf_bear and hma_cross_bear and vol_ok and vol_regime_ok and rsi_ok_short:
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