#!/usr/bin/env python3
"""
Experiment #795: 6h Primary + 12h/1d HTF — Volume-Confirmed HMA Trend with RSI Pullback

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments per instructions). 
This strategy combines proven elements from #791 (Sharpe=0.108 positive) with 
volume confirmation filter that most strategies ignore. Volume spikes often 
precede meaningful trend continuations after pullbacks.

Key innovations:
1. 1d HMA(21) for major trend bias (slow, reliable)
2. 12h HMA(16/48) for intermediate trend confirmation
3. 6h RSI(14) pullback entries (45/55 thresholds for trade generation)
4. Volume spike confirmation (vol > 1.5x 20-bar MA) — filters false breakouts
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (balanced for trade generation):
- LONG: 1d HMA bull + 12h HMA bull + RSI<50 + volume spike
- SHORT: 1d HMA bear + 12h HMA bear + RSI>50 + volume spike

Target: Sharpe>0.40 (beat current 6h best 0.399), trades>=30 train, trades>=3 test
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_rsi_vol_confirm_12h1d_v1"
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

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ma(volume, period=20):
    """Volume moving average for spike detection"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_12h_16_raw = calculate_hma(df_12h['close'].values, period=16)
    hma_12h_48_raw = calculate_hma(df_12h['close'].values, period=48)
    hma_12h_16_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_16_raw)
    hma_12h_48_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_48_raw)
    
    # Calculate 6h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    
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
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_12h_16_aligned[i]) or np.isnan(hma_12h_48_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h HMA TREND ===
        htf_12h_bull = hma_12h_16_aligned[i] > hma_12h_48_aligned[i]
        htf_12h_bear = hma_12h_16_aligned[i] < hma_12h_48_aligned[i]
        
        # === 6h HMA TREND ===
        hma_6h_bull = hma_16[i] > hma_48[i]
        hma_6h_bear = hma_16[i] < hma_48[i]
        
        # === RSI CONDITIONS (balanced for trade generation) ===
        rsi_pullback_long = rsi_14[i] < 50.0
        rsi_pullback_short = rsi_14[i] > 50.0
        rsi_extreme_oversold = rsi_14[i] < 35.0
        rsi_extreme_overbought = rsi_14[i] > 65.0
        
        # === VOLUME SPIKE CONFIRMATION ===
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # === ENTRY LOGIC (volume-confirmed pullbacks) ===
        desired_signal = 0.0
        
        # LONG: All HTF bull + RSI pullback + volume spike
        if htf_1d_bull and htf_12h_bull:
            if rsi_pullback_long and vol_spike:
                if rsi_extreme_oversold:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            elif hma_6h_bull and vol_spike:
                # HMA crossover confirmation with volume
                if i > 0 and not np.isnan(hma_16[i-1]) and not np.isnan(hma_48[i-1]):
                    if hma_16[i-1] <= hma_48[i-1] and hma_16[i] > hma_48[i]:
                        desired_signal = SIZE_STRONG
        
        # SHORT: All HTF bear + RSI pullback + volume spike
        elif htf_1d_bear and htf_12h_bear:
            if rsi_pullback_short and vol_spike:
                if rsi_extreme_overbought:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            elif hma_6h_bear and vol_spike:
                # HMA crossover confirmation with volume
                if i > 0 and not np.isnan(hma_16[i-1]) and not np.isnan(hma_48[i-1]):
                    if hma_16[i-1] >= hma_48[i-1] and hma_16[i] < hma_48[i]:
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