#!/usr/bin/env python3
"""
Experiment #021: 1d Donchian Breakout + Williams %R + Weekly Trend Filter

HYPOTHESIS: On 1d timeframe, 20-period Donchian breakouts capture major institutional
moves when confirmed by volume and aligned with weekly trend. 1d naturally limits
trade frequency to ~15-40 trades/year, avoiding fee drag. Williams %R provides 
momentum confirmation without adding complexity. Works in both directions using
1w HMA for trend bias.

TIMEFRAME: 1d primary
HTF: 1w for trend bias
TARGET: 50-120 total trades over 4 years (12-30/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_williams_1w_v1"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_williams_percent_r(high, low, close, period=14):
    """Williams %R momentum indicator"""
    n = len(close)
    wsr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        if highest != lowest:
            wsr[i] = -100 * (highest - close[i]) / (highest - lowest)
    
    return wsr

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate local 1d indicators
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Williams %R for momentum
    wsr = calculate_williams_percent_r(high, low, close, period=14)
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for additional confirmation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    stop_price = 0.0
    last_trade_bar = -100  # Prevent rapid re-entry
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY TREND FILTER ===
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        
        # === DONCHIAN STATE ===
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        
        # Previous close for breakout detection
        prev_close = close[i - 1] if i > 0 else close[0]
        prev_upper = donch_upper[i - 1] if i > 0 else donch_upper[i]
        prev_lower = donch_lower[i - 1] if i > 0 else donch_lower[i]
        
        # Breakout detection (close crosses channel)
        breakout_up = (close[i] > prev_upper) and (prev_close <= prev_upper)
        breakout_down = (close[i] < prev_lower) and (prev_close >= prev_lower)
        
        # === MOMENTUM (Williams %R) ===
        wsr_val = wsr[i] if not np.isnan(wsr[i]) else -50
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        min_bars_since_last_trade = i - last_trade_bar > 5
        
        if not in_position and min_bars_since_last_trade:
            # === NEW LONG ENTRY ===
            # Price breaks above Donchian upper + bullish weekly trend + momentum rising
            if breakout_up or price_above_upper:
                if price_above_1w_hma and (wsr_val > -80 or vol_spike):
                    desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Price breaks below Donchian lower + bearish weekly trend + momentum falling
            if breakout_down or price_below_lower:
                if not price_above_1w_hma and (wsr_val < -20 or vol_spike):
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (3 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT LOGIC ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price below lower channel OR RSI overbought OR trend flips
            if price_below_lower:
                exit_triggered = True
            if rsi[i] > 75:
                exit_triggered = True
            if not price_above_1w_hma and rsi[i] > 60:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price above upper channel OR RSI oversold OR trend flips
            if price_above_upper:
                exit_triggered = True
            if rsi[i] < 25:
                exit_triggered = True
            if price_above_1w_hma and rsi[i] < 40:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
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
                entry_bar = i
                last_trade_bar = i
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
            else:
                # Same direction - maintain position
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_bar = 0
        
        signals[i] = desired_signal
    
    return signals