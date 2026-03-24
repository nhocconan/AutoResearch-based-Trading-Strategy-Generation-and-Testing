#!/usr/bin/env python3
"""
Experiment #149: 4h Primary + 1d HTF — KAMA Trend + RSI Pullback + Volume Confirmation

Hypothesis: After analyzing 148 failed experiments, the pattern is clear:
- Complex regime switching (Choppiness, dual-regime) adds lag and blocks trades
- 12h timeframe showed promise (#142: +19.9% return) but 4h is required for this experiment
- KAMA (Kaufman Adaptive MA) adapts to volatility better than HMA/EMA in choppy markets
- Volume confirmation is MISSING from most failed strategies — adds edge
- LOOSE entry conditions are CRITICAL — many strategies got 0 trades (Sharpe=0.000)
- 1d HTF should provide BIAS not BLOCK — don't require perfect alignment

Key design choices:
- Timeframe: 4h (required by experiment, 20-50 trades/year target)
- HTF: 1d KAMA for major trend bias (not strict filter)
- Entry: KAMA(21) trend + RSI(14) pullback + Volume spike (1.5x avg)
- Position size: 0.30 (30% of capital, discrete levels)
- Stoploss: 2.5x ATR trailing
- LOOSE filters: RSI 25-75 range, volume > 1.3x average (not 2x)
- Ensure >=30 trades on train, >=3 on test by loosening conditions

Target: Sharpe>0.351 (beat current best), DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market noise — smooth in chop, responsive in trends
    From Perry Kaufman's "Trading Systems and Methods"
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    # Efficiency Ratio (ER): measures trend vs noise
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Initialize KAMA as SMA of first period
    kama[period] = np.mean(close[:period+1])
    
    for i in range(period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i-1]
        else:
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

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
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA for major trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
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
        if np.isnan(kama_4h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d KAMA) — LOOSE, provides direction not block ===
        htf_bull = close[i] > kama_1d_aligned[i]
        htf_bear = close[i] < kama_1d_aligned[i]
        
        # === 4h KAMA TREND ===
        kama_bull = close[i] > kama_4h[i]
        kama_bear = close[i] < kama_4h[i]
        
        # === KAMA SLOPE (trend strength) ===
        kama_slope_bull = kama_4h[i] > kama_4h[i-5] if i >= 5 else False
        kama_slope_bear = kama_4h[i] < kama_4h[i-5] if i >= 5 else False
        
        # === VOLUME CONFIRMATION (LOOSE: 1.3x average) ===
        vol_ratio = volume[i] / vol_sma[i] if vol_sma[i] > 1e-10 else 1.0
        vol_confirmed = vol_ratio > 1.3
        
        # === RSI PULLBACK (LOOSE: 25-75 range for entries) ===
        rsi_ok_long = rsi[i] > 25.0 and rsi[i] < 75.0
        rsi_ok_short = rsi[i] > 25.0 and rsi[i] < 75.0
        rsi_pullback_long = rsi[i] < 55.0  # pulling back in uptrend
        rsi_pullback_short = rsi[i] > 45.0  # pulling back in downtrend
        
        # === DESIRED SIGNAL (Simple trend + pullback + volume) ===
        desired_signal = 0.0
        
        # LONG: 4h KAMA bull + slope up + RSI pullback + volume OR HTF bull
        if kama_bull and kama_slope_bull and rsi_pullback_long:
            if vol_confirmed or htf_bull:
                desired_signal = SIZE
        
        # SHORT: 4h KAMA bear + slope down + RSI pullback + volume OR HTF bear
        if kama_bear and kama_slope_bear and rsi_pullback_short:
            if vol_confirmed or htf_bear:
                desired_signal = -SIZE
        
        # Fallback: Strong KAMA trend alone (ensure trades generate)
        if desired_signal == 0.0:
            if kama_bull and kama_slope_bull and rsi[i] > 30.0 and rsi[i] < 70.0:
                desired_signal = SIZE * 0.7
            elif kama_bear and kama_slope_bear and rsi[i] > 30.0 and rsi[i] < 70.0:
                desired_signal = -SIZE * 0.7
        
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
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.7
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.7
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