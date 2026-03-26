#!/usr/bin/env python3
"""
Experiment #003: 4h Primary + 1d HTF — Simple Volume-Confirmed Donchian

HYPOTHESIS:
Volume is the ONLY indicator that reliably distinguishes real breakouts from false moves.
Price breaks Donchian channel with above-average volume = institutional money is moving.
Combined with 1d HMA trend filter for direction, this captures the core momentum pattern.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- In bull markets: breakouts succeed more often, trend-following works
- In bear markets: breakdown volume spikes precede drops, short signals work
- Volume spike is symmetric - it confirms direction regardless of market bias
- Simple 2-condition entry (Donchian + Volume) = fewer false signals than multi-indicator

FAILURE ANALYSIS FROM THIS SESSION:
- All 20+ failed strategies stacked too many conditions (Fisher+RSI+Chop+Donchian = noise)
- Donchian-only strategies failed because volume confirms which breakouts are real
- The winning DB strategies all have volume confirmation

KEY INSIGHT: "Volume precedes price" - institutional orders show in volume BEFORE price moves.

Target: 75-200 total trades over 4 years | Sharpe>0.6 | DD>-35%
Size: 0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_volume_simple_1d_v2"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_volume_ratio(volume, taker_buy_volume, period=20):
    """
    Volume ratio = taker buy volume / total volume
    > 0.55 = bullish volume pressure
    < 0.45 = bearish volume pressure
    """
    n = len(volume)
    if n < period + 1:
        return np.full(n, np.nan)
    
    vol_ratio = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        total_vol = volume[i]
        buy_vol = taker_buy_volume[i]
        
        if total_vol > 0 and not np.isnan(buy_vol):
            vol_ratio[i] = buy_vol / total_vol
    
    return vol_ratio

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume for comparison"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load 1d HTF ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    vol_ratio = calculate_volume_ratio(volume, taker_buy, period=20)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30  # discrete position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1d TREND DIRECTION ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT CHECK ===
        # Breakout only valid if previous bar's close was below resistance (for long)
        # or above support (for short)
        prev_close_valid = i > 0 and not np.isnan(close[i-1])
        donch_breakout_long = False
        donch_breakout_short = False
        
        if prev_close_valid:
            donch_breakout_long = close[i] > donch_upper[i-1] and close[i-1] <= donch_upper[i-1]
            donch_breakout_short = close[i] < donch_lower[i-1] and close[i-1] >= donch_lower[i-1]
        
        # === VOLUME CONFIRMATION ===
        # Volume ratio > 0.52 = bullish pressure, < 0.48 = bearish
        vol_bullish = vol_ratio[i] > 0.52
        vol_bearish = vol_ratio[i] < 0.48
        
        # Additional: current volume above recent average
        vol_above_avg = volume[i] > vol_sma[i] if not np.isnan(vol_sma[i]) else False
        
        # === ENTRY LOGIC: Donchian + Volume + 1d Trend ===
        # SIMPLE: 3 conditions max
        desired_signal = 0.0
        
        # LONG: Breakout + Volume + 1d bullish
        if price_above_1d and donch_breakout_long and vol_bullish:
            desired_signal = SIZE
        
        # SHORT: Breakdown + Volume + 1d bearish
        elif price_below_1d and donch_breakout_short and vol_bearish:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL ===
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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