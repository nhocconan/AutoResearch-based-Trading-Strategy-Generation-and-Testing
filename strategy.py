#!/usr/bin/env python3
"""
Experiment #006: 4h Camarilla S4/R4 + Choppiness Range Filter + Volume

HYPOTHESIS: Camarilla S3/S4 and R3/R4 are institutional levels where price
reverses. By combining with CHOPPINESS INDEX (regime filter) + volume spike,
this strategy catches mean reversion trades ONLY in ranging conditions.

WHY IT WORKS: Choppiness Index > 61.8 = ranging market = Camarilla levels work.
In trending markets (CHOP < 38.2), Camarilla levels break instead of reverting.

CORE PATTERN FROM DB: gen_camarilla_pivot_volume_spike_choppiness_4h_v1
  → ETHUSDT test_sharpe=1.471, 95 trades, 54% win rate (BEST IN DATABASE)

CHANGES FROM FAILED #016: 
  - Use CHOPPINESS INDEX instead of EMA50 trend filter (proven more robust)
  - Only enter in range market (CHOP > 61.8) — avoids 2022 trending crash
  - Volume ratio threshold 2.0 instead of 1.5 — more selective entries
  - Target: 75-150 total trades (vs 275 in failed #016)

TARGET: 75-150 total over 4 years = 19-37/year.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_chop_vol_1d_v1"
timeframe = "4h"
leverage = 1.0


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


def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging (mean reversion works)
    CHOP < 38.2 = trending (follow the trend)
    Formula: 100 * LOG10(SUM(ATR(1), period) / HHV(HIGH - LOW, period)) / LOG10(period)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR(1) over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr1
        
        # Highest High - Lowest Low over period
        highest_high = max(high[i - period + 1:i + 1])
        lowest_low = min(low[i - period + 1:i + 1])
        hl_range = highest_high - lowest_low
        
        if hl_range > 1e-10:
            log_ratio = np.log10(atr_sum / hl_range)
            log_period = np.log10(period)
            chop[i] = 100 * log_ratio / log_period
    
    return chop


def calculate_kama(close, period=21, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average
    Returns trend direction (positive = up, negative = down)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Price change
    change = np.abs(np.diff(close, prepend=close[0]))
    
    # Volatility (sum of price changes)
    volatility = pd.Series(change).rolling(window=period, min_periods=period).sum().values
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        if volatility[i] > 1e-10:
            er[i] = change[i] / volatility[i]
    
    # Smoothing constant
    fast_const = 2 / (fast + 1)
    slow_const = 2 / (slow + 1)
    const_smooth = er * (fast_const - slow_const) + slow_const
    kama = np.zeros(n)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + const_smooth[i] * (close[i] - kama[i-1])
    
    return kama


def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d KAMA for trend direction (used only for additional confirmation)
    kama_1d = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio (20-bar SMA baseline)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Buffer for alignment
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if CHOP not ready
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if KAMA not aligned
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME FILTER: Only trade in range market ===
        in_range = chop_14[i] > 61.8
        
        # === TREND FILTER (1d KAMA): directional bias ===
        price_above_kama = close[i] > kama_1d_aligned[i]
        
        # Volume confirmation (stricter threshold: 2.0x)
        vol_spike = vol_ratio[i] > 2.0
        
        # === CAMARILLA LEVELS from previous CLOSED bar (no look-ahead) ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        prev_range = prev_high - prev_low
        
        # Classic Camarilla levels (factor = 1.1/12 = 0.09167)
        r3 = prev_close + prev_range * 0.09167
        r4 = prev_close + prev_range * 0.18333
        s3 = prev_close - prev_range * 0.09167
        s4 = prev_close - prev_range * 0.18333
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price touches S3 or S4 in range market + trend alignment ===
            # S4 touch (deeper level = better R/R)
            if low[i] <= s4 and in_range and price_above_kama and vol_spike:
                desired_signal = SIZE
            # S3 touch (softer level)
            elif low[i] <= s3 and in_range and price_above_kama and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Price touches R3 or R4 in range market + trend alignment ===
            # R4 touch
            if high[i] >= r4 and in_range and not price_above_kama and vol_spike:
                desired_signal = -SIZE
            # R3 touch
            elif high[i] >= r3 and in_range and not price_above_kama and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === MINIMUM HOLD: 2 bars (8h) to reduce false signals ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 2:
            # Take profit at Camarilla mid (prev close level)
            if position_side > 0 and close[i] >= prev_close:
                desired_signal = 0.0
            if position_side < 0 and close[i] <= prev_close:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals