#!/usr/bin/env python3
"""
Experiment #258: 4h Primary + 1d HTF — Vol Spike Mean Reversion + HTF Trend Filter

Hypothesis: After 230+ failed experiments, regime-switching strategies are overfitted.
Simple vol-spike mean reversion with HTF trend filter should work better:

1. VOL SPIKE REVERSION: ATR(7)/ATR(30) > 2.0 indicates panic/extreme moves
   - After panic, price tends to revert (vol crush)
   - Entry when price at BB extreme + vol spike

2. HTF TREND FILTER: Only trade with 1d HMA(50) direction
   - Long only when close > 1d HMA(50)
   - Short only when close < 1d HMA(50)
   - Reduces counter-trend trades that fail in 2022 crash

3. RSI CONFIRMATION: RSI(14) < 30 for long, > 70 for short
   - Ensures we're catching extremes, not mid-range noise

4. VOLUME CONFIRMATION: Volume > 1.5x 20-bar avg
   - Panic moves have high volume, confirms the spike

5. ASYMMETRIC SIZING: 0.25 base, 0.30 when all 4 conditions align
   - Reduces exposure during uncertain conditions

Why this should work:
- Vol spike reversion has proven edge in crypto (panic = buying opportunity)
- HTF filter prevents disaster in 2022-style crashes
- Simple logic = fewer whipsaws than regime-switching
- 4h TF = 20-50 trades/year target (not too many fees)

Target: Sharpe>0.45, DD>-35%, trades>=25 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volspike_mr_hma_1d_v1"
timeframe = "4h"
leverage = 1.0

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

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Bollinger Bands with wider std for extreme detection"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower

def calculate_vol_spike_ratio(high, low, close, short_period=7, long_period=30):
    """ATR ratio to detect volatility spikes"""
    n = len(close)
    if n < long_period + 1:
        return np.full(n, np.nan)
    
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    ratio = np.zeros(n)
    ratio[:] = np.nan
    for i in range(long_period, n):
        if atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def calculate_volume_ratio(volume, period=20):
    """Volume relative to moving average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    ratio = np.zeros(n)
    ratio[:] = np.nan
    for i in range(period, n):
        if vol_ma[i] > 1e-10:
            ratio[i] = volume[i] / vol_ma[i]
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.5)
    vol_ratio = calculate_vol_spike_ratio(high, low, close, short_period=7, long_period=30)
    volume_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
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
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]) or np.isnan(volume_ratio[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === VOL SPIKE DETECTION ===
        vol_spike = vol_ratio[i] > 2.0  # ATR(7) > 2x ATR(30)
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume_ratio[i] > 1.5  # Volume > 1.5x average
        
        # === PRICE AT BB EXTREME ===
        at_bb_lower = close[i] <= bb_lower[i]
        at_bb_upper = close[i] >= bb_upper[i]
        
        # === RSI EXTREME ===
        rsi_oversold = rsi[i] < 30.0
        rsi_overbought = rsi[i] > 70.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        signal_strength = 0
        
        # LONG: Vol spike + BB lower + RSI oversold + HTF bull + Volume confirmed
        long_conditions = 0
        if vol_spike:
            long_conditions += 1
        if at_bb_lower:
            long_conditions += 1
        if rsi_oversold:
            long_conditions += 1
        if htf_bull:
            long_conditions += 1
        if vol_confirmed:
            long_conditions += 1
        
        # Need at least 4 of 5 conditions for long
        if long_conditions >= 4 and htf_bull:
            if long_conditions >= 5:
                desired_signal = SIZE_STRONG
                signal_strength = 2
            else:
                desired_signal = SIZE_BASE
                signal_strength = 1
        
        # SHORT: Vol spike + BB upper + RSI overbought + HTF bear + Volume confirmed
        short_conditions = 0
        if vol_spike:
            short_conditions += 1
        if at_bb_upper:
            short_conditions += 1
        if rsi_overbought:
            short_conditions += 1
        if htf_bear:
            short_conditions += 1
        if vol_confirmed:
            short_conditions += 1
        
        # Need at least 4 of 5 conditions for short
        if short_conditions >= 4 and htf_bear:
            if short_conditions >= 5:
                desired_signal = -SIZE_STRONG
                signal_strength = 2
            else:
                desired_signal = -SIZE_BASE
                signal_strength = 1
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
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