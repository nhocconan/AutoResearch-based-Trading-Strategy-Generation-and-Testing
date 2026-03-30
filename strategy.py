#!/usr/bin/env python3
"""
Experiment #025: 4h KAMA Trend + Donchian Breakout + RSI Confirmation

HYPOTHESIS: 1w KAMA(10) provides institutional-grade trend direction that 
doesn't flip on noise. Combined with 4h Donchian(20) breakout for structure
and RSI(14) for momentum confirmation, this creates tight entries that avoid
whipsaws. Choppiness filter avoids range-bound markets.

WHY IT WORKS IN BOTH BULL AND BEAR:
- Bull: KAMA rising + price > KAMA + Donchian breakout + RSI > 50 = strong long
- Bear: KAMA falling + price < KAMA + Donchian breakdown + RSI < 50 = strong short
- Range: Choppiness > 61.8 = no trades (avoids 2022 crash whipsaws)

TARGET: 75-150 total trades over 4 years (18-37/year).
Signal size: 0.30 (discrete levels).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_donchian_rsi_chop_1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n, dtype=np.float64)
    direction = np.abs(close[period:] - close[:-period])
    volatility = np.zeros(n - period, dtype=np.float64)
    
    for i in range(period, n):
        sum_val = 0.0
        for j in range(i - period + 1, i + 1):
            sum_val += abs(close[j] - close[j - 1])
        volatility[i - period] = sum_val
    
    for i in range(len(er) - period):
        if volatility[i] > 1e-10:
            er[i + period] = direction[i] / volatility[i]
    
    # Smooth constant
    fast_const = 2 / (fast + 1)
    slow_const = 2 / (slow + 1)
    square_slow = slow_const * slow_const
    
    kama = np.full(n, np.nan)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_const - slow_const) + slow_const) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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
    """RSI indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - >61.8 = choppy, <38.2 = trending"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr_j = high[j] - low[j] if j == 0 else max(high[j] - low[j], abs(high[j] - close[j-1]))
            atr_sum += tr_j
        
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0:
            chop[i] = 100 * (np.log10(atr_sum / range_sum) / np.log10(period + 1))
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w KAMA for institutional trend (call ONCE before loop)
    kama_1w = calculate_kama(df_1w['close'].values, period=10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_up, donchian_lo = calculate_donchian(high, low, period=20)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 200  # Need enough for KAMA + Donchian + Choppiness
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === 1w KAMA TREND FILTER ===
        kama_val = kama_1w_aligned[i]
        kama_trend_up = close[i] > kama_val
        kama_trend_down = close[i] < kama_val
        
        # KAMA slope (compare to 5 bars ago)
        kama_slope_up = kama_val > kama_1w_aligned[i - 5] if i >= 5 else False
        kama_slope_down = kama_val < kama_1w_aligned[i - 5] if i >= 5 else False
        
        # === RSI MOMENTUM ===
        rsi = rsi_14[i]
        rsi_bullish = rsi > 50 if not np.isnan(rsi) else False
        rsi_bearish = rsi < 50 if not np.isnan(rsi) else False
        
        # === CHOPPINESS REGIME ===
        chop = chop_14[i]
        is_choppy = chop > 61.8 if not np.isnan(chop) else False
        is_trending = chop < 50 if not np.isnan(chop) else True
        
        # === DONCHIAN BREAKOUT (use shift(1) to avoid look-ahead) ===
        donchian_broken_up = close[i] > donchian_up[i - 1] if i > 0 else False
        donchian_broken_down = close[i] < donchian_lo[i - 1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Conditions: KAMA rising + price > KAMA + Donchian breakout + RSI > 50 + volume
            if kama_trend_up and kama_slope_up:
                if donchian_broken_up and rsi_bullish and vol_spike and not is_choppy:
                    desired_signal = SIZE
            # Fallback: Strong momentum with RSI + volume in uptrend
            elif kama_trend_up and rsi > 60 and vol_spike:
                if donchian_broken_up:
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Conditions: KAMA falling + price < KAMA + Donchian breakdown + RSI < 50 + volume
            if kama_trend_down and kama_slope_down:
                if donchian_broken_down and rsi_bearish and vol_spike and not is_choppy:
                    desired_signal = -SIZE
            # Fallback: Strong momentum with RSI + volume in downtrend
            elif kama_trend_down and rsi < 40 and vol_spike:
                if donchian_broken_down:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position:
            bars_held = i - entry_bar
            
            if position_side > 0:
                # Long: stop if price drops below entry - 2.5 ATR
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                # Exit if KAMA flips or RSI deteriorates severely
                elif (not kama_trend_up and bars_held >= 4) or (rsi < 35 and bars_held >= 2):
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short: stop if price rises above entry + 2.5 ATR
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                # Exit if KAMA flips or RSI improves severely
                elif (not kama_trend_down and bars_held >= 4) or (rsi > 65 and bars_held >= 2):
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 2 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 2:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals