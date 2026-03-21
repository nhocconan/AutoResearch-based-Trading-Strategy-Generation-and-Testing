#!/usr/bin/env python3
"""
Experiment #369: 1h KAMA Adaptive Trend + 4h KAMA Bias + MACD Momentum + Volume Filter + ATR Stop
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better than HMA/EMA.
During high volatility (trending), KAMA follows price closely. During low volatility (ranging), KAMA flattens.
This adaptiveness should reduce whipsaws in 2022 crash and 2025 bear market compared to fixed-period EMAs.
4h KAMA provides intermediate trend bias (more responsive than daily for 1h entries).
MACD histogram confirms momentum direction. Volume filter ensures breakout validity.
ATR(14) stoploss at 2.0x protects capital during reversals.
Timeframe: 1h (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with proper trade frequency (loose RSI 30-70 thresholds).
Key insight: KAMA's efficiency ratio adapts to regime automatically - no need for separate chop filter.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_4h_kama_macd_volume_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.abs(close[0:period] - close[0])
    
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        if i >= period:
            volatility[i] -= np.abs(close[i-period] - close[i-period-1])
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = sc ** 2
    
    # Calculate KAMA
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_macd(close, fast=12, slow=26, signal_period=9):
    """Calculate MACD line, signal line, and histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, min_periods=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    kama_4h = calculate_kama(df_4h['close'].values, period=10)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Calculate 1h indicators
    kama_1h = calculate_kama(close, period=10)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    vol_ma = calculate_volume_ma(volume, 20)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(kama_1h[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (SOFT filter - boosts confidence)
        htf_bullish = not np.isnan(kama_4h_aligned[i]) and close[i] > kama_4h_aligned[i]
        htf_bearish = not np.isnan(kama_4h_aligned[i]) and close[i] < kama_4h_aligned[i]
        
        # 1h KAMA trend
        kama_bullish = close[i] > kama_1h[i]
        kama_bearish = close[i] < kama_1h[i]
        
        # KAMA slope (trend strength)
        kama_slope_long = kama_1h[i] > kama_1h[i-5] if i >= 5 else False
        kama_slope_short = kama_1h[i] < kama_1h[i-5] if i >= 5 else False
        
        # MACD momentum confirmation
        macd_bullish = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1] if i >= 1 else macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1] if i >= 1 else macd_hist[i] < 0
        
        # Volume confirmation (above average)
        volume_ok = not np.isnan(vol_ma[i]) and volume[i] > 0.8 * vol_ma[i]
        
        # RSI filter (LOOSE thresholds to ensure trade frequency)
        rsi_ok_long = rsi[i] > 30 and rsi[i] < 80  # Not oversold, room to run
        rsi_ok_short = rsi[i] < 70 and rsi[i] > 20  # Not overbought, room to fall
        
        # RSI momentum
        rsi_momentum_long = rsi[i] > 45
        rsi_momentum_short = rsi[i] < 55
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: 1h KAMA bullish + 4h KAMA bullish + MACD bullish + RSI ok
        if kama_bullish and htf_bullish and macd_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: 1h KAMA bullish + KAMA slope up + MACD bullish + volume ok
        elif kama_bullish and kama_slope_long and macd_bullish and volume_ok:
            new_signal = SIZE_ENTRY
        # Tertiary: 1h KAMA bullish + 4h KAMA bullish + RSI momentum (MACD neutral ok)
        elif kama_bullish and htf_bullish and rsi_momentum_long and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Quaternary: KAMA cross above (price crosses KAMA) + volume + RSI ok
        elif kama_bullish and close[i-1] <= kama_1h[i-1] and volume_ok and rsi[i] > 35:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: 1h KAMA bearish + 4h KAMA bearish + MACD bearish + RSI ok
        if kama_bearish and htf_bearish and macd_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: 1h KAMA bearish + KAMA slope down + MACD bearish + volume ok
        elif kama_bearish and kama_slope_short and macd_bearish and volume_ok:
            new_signal = -SIZE_ENTRY
        # Tertiary: 1h KAMA bearish + 4h KAMA bearish + RSI momentum (MACD neutral ok)
        elif kama_bearish and htf_bearish and rsi_momentum_short and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Quaternary: KAMA cross below (price crosses KAMA) + volume + RSI ok
        elif kama_bearish and close[i-1] >= kama_1h[i-1] and volume_ok and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals