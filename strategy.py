#!/usr/bin/env python3
"""
Experiment #010: 1d Donchian Breakout + 1w HMA Trend + Volume + Chop Regime

HYPOTHESIS: Daily Donchian(20) breakouts capture major trend moves. When aligned with
weekly HMA(21) trend bias, confirmed by volume spike, and in trending regime (low chop),
these breakouts have high probability of continuation.

WHY THIS WORKS IN BOTH BULL AND BEAR:
- Bull: Long breakouts when 1w HMA bullish (trend continuation)
- Bear: Short breakouts when 1w HMA bearish (trend continuation)
- Volume confirms institutional participation, not fake breakouts
- Chop filter avoids range-bound whipsaws

TARGET: 40-100 total trades over 4 years (10-25/year) - VERY TIGHT entries
DB reference: mtf_1d_kama_rsi_chop_regime_1w_v1 (Sharpe=1.31, 74tr)

KEY FIXES from previous failures:
1. Only enter on BREAKOUT bar (price crosses level), not just being near level
2. Track previous signal to avoid churn (only flip when signal changes)
3. Minimum hold period: 3 bars before allowing exit
4. Tighter volume: 1.5x (not 1.3x)
5. Stricter chop: <50 (not <61.8)
6. 1d timeframe = naturally fewer trades than 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_hma_vol_chop_v1"
timeframe = "1d"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    We require CHOP < 50 for entries (trending regime)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel: upper = highest high, lower = lowest low over period"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume MA and ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_in_trade = 0
    prev_signal = 0.0
    min_hold_bars = 3
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position and bars_in_trade >= min_hold_bars:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position and bars_in_trade >= min_hold_bars:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position and bars_in_trade >= min_hold_bars:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position and bars_in_trade >= min_hold_bars:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        is_trending = chop < 50.0  # Stricter: only trending regime
        
        # === TREND BIAS (1w HMA) ===
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5  # Stricter: 1.5x
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Only enter on BREAKOUT bar (price crosses level from outside)
        prev_upper = donchian_upper[i-1] if i > 0 else donchian_upper[i]
        prev_lower = donchian_lower[i-1] if i > 0 else donchian_lower[i]
        
        # Long breakout: price crosses ABOVE upper (was below, now above)
        long_breakout = (high[i] > prev_upper) and (close[i] > prev_upper)
        
        # Short breakout: price crosses BELOW lower (was above, now below)
        short_breakout = (low[i] < prev_lower) and (close[i] < prev_lower)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if is_trending:
            # LONG: Donchian breakout + 1w HMA bullish + volume spike
            if long_breakout and price_above_1w_hma and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Donchian breakout + 1w HMA bearish + volume spike
            if short_breakout and (not price_above_1w_hma) and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
        
        # === TAKE PROFIT at opposite Donchian level ===
        tp_triggered = False
        if in_position and position_side > 0 and not np.isnan(donchian_lower[i]):
            if low[i] <= donchian_lower[i]:
                tp_triggered = True
        
        if in_position and position_side < 0 and not np.isnan(donchian_upper[i]):
            if high[i] >= donchian_upper[i]:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === MINIMUM HOLD PERIOD ===
        # Don't exit before min_hold_bars unless stoploss/TP hit
        if in_position and bars_in_trade < min_hold_bars:
            if desired_signal == 0.0 and not stoploss_triggered and not tp_triggered:
                desired_signal = prev_signal  # Keep previous signal
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                bars_in_trade = 0
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                bars_in_trade += 1
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                bars_in_trade = 0
        
        signals[i] = desired_signal
        prev_signal = desired_signal
    
    return signals