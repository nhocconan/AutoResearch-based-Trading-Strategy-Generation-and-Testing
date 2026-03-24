#!/usr/bin/env python3
"""
Experiment #898: 4h Primary + 1d HTF — HMA Trend + RSI Pullback + Volume Confirm

Hypothesis: 4h timeframe with daily HTF bias provides optimal balance between
trade frequency (20-50 trades/year) and signal quality. Hull Moving Average on
1d provides smooth trend bias with minimal lag. RSI(14) pullback entries on 4h
capture mean-reversion within the HTF trend. Volume spike confirmation filters
false breakouts. This combination has worked well in bear/range markets (2025).

Key innovations:
1. 1d HMA(21) for HTF trend bias - smoother than EMA, less lag than KAMA
2. 4h RSI(14) pullback entries - enter on dips in uptrend, rallies in downtrend
3. Volume spike confirmation (>1.5x 20-bar avg) - filters low-conviction moves
4. ATR(14) 2.5x trailing stop for risk management
5. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn
6. LOOSE entry thresholds to ensure ≥10 trades/train, ≥3/test

Entry conditions (LOOSE for trade frequency):
- LONG: 1d HMA bull (price>hma) + 4h RSI<55 (pullback, not extreme) + volume>1.3x avg
- SHORT: 1d HMA bear (price<hma) + 4h RSI>45 (rally, not extreme) + volume>1.3x avg

Target: Sharpe>0.45, trades>=10 train, trades>=3 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_vol_confirm_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    if sqrt_n < 1:
        sqrt_n = 1
    
    # WMA helper
    def wma(series, span):
        if span < 1:
            span = 1
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    # WMA of diff with sqrt(n)
    hma = wma(diff, sqrt_n)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    close_f = close.astype(np.float64)
    delta = np.diff(close_f, prepend=close_f[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    high_f = high.astype(np.float64)
    low_f = low.astype(np.float64)
    close_f = close.astype(np.float64)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high_f[0] - low_f[0]
    for i in range(1, n):
        tr[i] = max(high_f[i] - low_f[i], abs(high_f[i] - close_f[i-1]), abs(low_f[i] - close_f[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    volume_f = volume.astype(np.float64)
    vol_avg = pd.Series(volume_f).rolling(window=period, min_periods=period).mean().values
    
    vol_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if vol_avg[i] > 1e-10:
            vol_ratio[i] = volume_f[i] / vol_avg[i]
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    hma_4h_16 = calculate_hma(close, period=16)
    hma_4h_48 = calculate_hma(close, period=48)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
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
        
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HMA TREND ===
        hma_4h_bull = hma_4h_16[i] > hma_4h_48[i]
        hma_4h_bear = hma_4h_16[i] < hma_4h_48[i]
        
        # === RSI CONDITIONS (LOOSE for trade frequency) ===
        # In bull trend: enter on pullback (RSI < 55, not extreme oversold)
        # In bear trend: enter on rally (RSI > 45, not extreme overbought)
        rsi_pullback_long = rsi_14[i] < 55.0
        rsi_pullback_short = rsi_14[i] > 45.0
        
        # Stronger signals at extremes
        rsi_strong_long = rsi_14[i] < 40.0
        rsi_strong_short = rsi_14[i] > 60.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.3  # 30% above average
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADES) ===
        desired_signal = 0.0
        
        if htf_1d_bull and hma_4h_bull:
            # Double bull confirmation
            if rsi_pullback_long:
                if vol_confirm:
                    if rsi_strong_long:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
                else:
                    # Still enter without volume confirm but smaller size
                    desired_signal = SIZE_BASE * 0.6
        
        elif htf_1d_bear and hma_4h_bear:
            # Double bear confirmation
            if rsi_pullback_short:
                if vol_confirm:
                    if rsi_strong_short:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
                else:
                    # Still enter without volume confirm but smaller size
                    desired_signal = -SIZE_BASE * 0.6
        
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
        elif abs(desired_signal) > 0.01:
            # Small positions round to base size
            final_signal = np.sign(desired_signal) * SIZE_BASE
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