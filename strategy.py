#!/usr/bin/env python3
"""
Experiment #1147: 6h Primary + 1d HTF — HMA Trend + RSI Pullback + Volume Filter

Hypothesis: 6h timeframe sits between 4h (proven) and 12h (too slow). Using 1d HMA for 
trend direction and 6h RSI pullbacks for entry timing should capture multi-day swings 
with better risk/reward than pure trend-following. Volume filter reduces false breakouts.

Key innovations:
1. 1d HMA(21) for primary trend bias (aligned via mtf_data helper)
2. 6h RSI(14) pullback entries: long when RSI 35-50 in uptrend, short when 50-65 in downtrend
3. Volume confirmation: entry volume > 0.8 * 20-bar avg volume
4. ATR(14) 2.5x trailing stoploss
5. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn
6. 1d HMA slope filter: only trade when 1d HMA has clear direction

Why this should work:
- 6h captures 2-4 day swings (optimal for crypto perpetuals)
- RSI pullback entries avoid chasing breakouts (better fill prices)
- Volume filter ensures institutional participation
- 1d HTF bias prevents counter-trend trades in strong trends
- Simpler than regime-switching (fewer failure modes)

Entry conditions (LOOSE to guarantee trades):
- LONG: 1d_HMA slope > 0 + price > 1d_HMA + RSI(14) 35-50 + volume confirmed
- SHORT: 1d_HMA slope < 0 + price < 1d_HMA + RSI(14) 50-65 + volume confirmed

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_rsi_pullback_vol_1d_v1"
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

def calculate_sma(series, period):
    """Simple Moving Average"""
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    vol_sma20 = calculate_sma(volume, 20)
    
    # Calculate 1d HMA slope (rate of change over 3 bars)
    hma_1d_slope = np.full(n, np.nan, dtype=np.float64)
    for i in range(3, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-3]):
            hma_1d_slope[i] = (hma_1d_aligned[i] - hma_1d_aligned[i-3]) / hma_1d_aligned[i-3] * 100
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_sma20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1d_slope[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        hma_slope_bull = hma_1d_slope[i] > 0.5  # >0.5% slope over 3 bars
        hma_slope_bear = hma_1d_slope[i] < -0.5
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 0.8 * vol_sma20[i] if vol_sma20[i] > 0 else False
        
        # === ENTRY LOGIC (RSI PULLBACK IN TREND) ===
        desired_signal = 0.0
        
        # LONG: Uptrend + RSI pullback to 35-50 zone
        if hma_1d_bull and hma_slope_bull:
            if 35.0 <= rsi_14[i] <= 50.0 and vol_confirmed:
                desired_signal = SIZE_BASE
            elif 30.0 <= rsi_14[i] < 35.0 and vol_confirmed:
                desired_signal = SIZE_STRONG
        
        # SHORT: Downtrend + RSI pullback to 50-65 zone
        elif hma_1d_bear and hma_slope_bear:
            if 50.0 <= rsi_14[i] <= 65.0 and vol_confirmed:
                desired_signal = -SIZE_BASE
            elif 65.0 < rsi_14[i] <= 70.0 and vol_confirmed:
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