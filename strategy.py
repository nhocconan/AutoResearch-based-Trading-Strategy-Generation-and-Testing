#!/usr/bin/env python3
"""
Experiment #026: 1d Donchian Close Breakout + Volume + 1w HMA Trend

HYPOTHESIS: On 1d timeframe, waiting for price to CLOSE beyond Donchian(20) 
(never just touch) filters out failed breakouts. Volume spike confirms 
institutional conviction. 1w HMA alignment ensures we're trading WITH 
the larger trend. ATR(14) stoploss at 2.5x prevents large losses.

KEY DIFFERENCE FROM FAILED STRATS: 
- Previous 0-trade attempts required multiple conflicting conditions
- Previous overtrading attempts were too loose (267-343 trades)
- This: ONE clear signal (close beyond Donchian) + volume + 1w trend

TIMEFRAME: 1d primary
HTF: 1w for trend alignment only
TARGET: 60-120 total trades over 4 years (15-30/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_close_vol_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - vectorized"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    # Use pandas rolling for WMA calculation (faster than loop)
    half_wma = pd.Series(close).rolling(half, min_periods=half).apply(
        lambda x: np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    full_wma = pd.Series(close).rolling(period, min_periods=period).apply(
        lambda x: np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    
    diff = 2 * half_wma - full_wma
    
    result = pd.Series(diff).rolling(sqrt_n, min_periods=sqrt_n).apply(
        lambda x: np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    
    return result

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
    """Donchian Channel - upper and lower bands"""
    n = len(high)
    upper = pd.Series(high).rolling(period, min_periods=period).max().values
    lower = pd.Series(low).rolling(period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w HMA for trend direction
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Previous 1w HMA for trend direction change
    hma_1w_prev_raw = np.roll(calculate_hma(df_1w['close'].values, period=21), 1)
    hma_1w_prev_raw[0] = np.nan
    hma_1w_prev_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_prev_raw)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for momentum check
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
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1w HMA) ===
        # Uptrend: price above 1w HMA AND 1w HMA rising
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        1w_hma_rising = hma_1w_aligned[i] > hma_1w_prev_aligned[i] if not np.isnan(hma_1w_prev_aligned[i]) else True
        bull_trend = price_above_1w_hma and 1w_hma_rising
        
        # Downtrend: price below 1w HMA AND 1w HMA falling
        price_below_1w_hma = close[i] < hma_1w_aligned[i]
        1w_hma_falling = hma_1w_aligned[i] < hma_1w_prev_aligned[i] if not np.isnan(hma_1w_prev_aligned[i]) else False
        bear_trend = price_below_1w_hma and 1w_hma_falling
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === RSI VALUE ===
        rsi_val = rsi[i]
        
        # === DONCHIAN CLOSE BREAKOUT (strict - must CLOSE beyond, not touch) ===
        # Previous bar's channel
        prev_upper = donch_upper[i-1] if i > 0 else donch_upper[i]
        prev_lower = donch_lower[i-1] if i > 0 else donch_lower[i]
        
        # Current bar CLOSES above previous upper band
        close_above_prev_upper = close[i] > prev_upper
        # Current bar CLOSES below previous lower band
        close_below_prev_lower = close[i] < prev_lower
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Price CLOSES above previous upper Donchian + volume spike + bull trend
            if close_above_prev_upper and vol_spike and bull_trend:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Price CLOSES below previous lower Donchian + volume spike + bear trend
            if close_below_prev_lower and vol_spike and bear_trend:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
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
        
        # === EXIT LOGIC ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price CLOSES below lower band OR RSI < 30
            if close[i] < donch_lower[i]:
                exit_triggered = True
            if rsi_val < 30:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price CLOSES above upper band OR RSI > 70
            if close[i] > donch_upper[i]:
                exit_triggered = True
            if rsi_val > 70:
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
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            # else: maintain position
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals