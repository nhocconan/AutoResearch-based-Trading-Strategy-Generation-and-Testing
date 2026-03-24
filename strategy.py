#!/usr/bin/env python3
"""
Experiment #024: 4h Primary + 12h HTF — HMA Trend + RSI Pullback + Volume

Hypothesis: After 23 failed experiments, the key issue is OVER-FILTERING.
Strategies with ADX, Choppiness, Connors RSI, etc. generate 0 trades because
conditions never align. This strategy uses PROVEN SIMPLE logic:

1. HMA(21) trend on 4h - faster than EMA, less lag
2. HMA(21) trend on 12h - HTF bias for direction
3. RSI(14) pullback - enter on dips in uptrend, rallies in downtrend
4. Volume confirmation - avoid low-volume false moves
5. NO ADX, NO Choppiness, NO complex regime filters

Entry Logic (LOOSE to ensure trades):
- Long: 4h close > 4h HMA + 12h close > 12h HMA + RSI(14) between 25-55 + vol > avg
- Short: 4h close < 4h HMA + 12h close < 12h HMA + RSI(14) between 45-75 + vol > avg

Why this should work:
- HMA is proven (current best uses HMA)
- RSI pullback = buy dips, not breakouts (more trades)
- Volume filter = avoid false signals
- Loose RSI range (25-55 long, 45-75 short) = ensures entries trigger
- No conflicting filters = more trade opportunities

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Size: 0.30 (discrete, minimizes fee churn)
Target: Sharpe>0.3, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_vol_pullback_loose_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) window
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate WMA for period/2
    half_period = int(period / 2)
    wma_half = np.full(n, np.nan)
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        wma_half[i] = np.sum(close[i - half_period + 1:i + 1] * weights) / np.sum(weights)
    
    # Calculate WMA for full period
    wma_full = np.full(n, np.nan)
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma_full[i] = np.sum(close[i - period + 1:i + 1] * weights) / np.sum(weights)
    
    # Calculate raw HMA
    raw_hma = 2.0 * wma_half - wma_full
    
    # Smooth with WMA of sqrt(period)
    sqrt_period = int(np.sqrt(period))
    hma = np.full(n, np.nan)
    for i in range(sqrt_period - 1, n):
        if np.isnan(raw_hma[i]):
            hma[i] = np.nan
        else:
            weights = np.arange(1, sqrt_period + 1)
            start_idx = max(0, i - sqrt_period + 1)
            valid_raw = raw_hma[start_idx:i + 1]
            valid_weights = weights[-len(valid_raw):]
            if len(valid_raw) > 0 and not np.all(np.isnan(valid_raw)):
                hma[i] = np.nansum(valid_raw * valid_weights) / np.sum(valid_weights)
    
    return hma

def calculate_rsi(close, period=14):
    """RSI - momentum indicator for pullback entries"""
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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_avg(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_avg[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_avg

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for HTF trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size
    
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
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HMA TREND ALIGNMENT (4h + 12h) ===
        hma_4h_bull = close[i] > hma_4h[i]
        hma_4h_bear = close[i] < hma_4h[i]
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_ok = volume[i] > vol_avg[i] * 0.8  # At least 80% of avg volume
        
        # === RSI PULLBACK (LOOSE thresholds to ensure trades) ===
        # Long: RSI pulled back but not oversold (25-55 range)
        rsi_ok_long = 25.0 <= rsi[i] <= 55.0
        # Short: RSI rallied but not overbought (45-75 range)
        rsi_ok_short = 45.0 <= rsi[i] <= 75.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry: Dual HMA bullish + RSI pullback + Volume
        if hma_4h_bull and hma_12h_bull and rsi_ok_long and vol_ok:
            desired_signal = SIZE
        
        # Short entry: Dual HMA bearish + RSI pullback + Volume
        elif hma_4h_bear and hma_12h_bear and rsi_ok_short and vol_ok:
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