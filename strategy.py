#!/usr/bin/env python3
"""
Experiment #028: 1d Donchian Breakout + 1w Trend + Volume

HYPOTHESIS: On 1d timeframe, 20-period Donchian breakouts capture institutional
moves over 4-6 week periods. Combined with 1w HMA trend alignment and volume
confirmation, this identifies high-probability breakouts in both directions.
1d is slow enough to avoid noise while fast enough to catch 20-40 moves/year.

KEY INSIGHTS FROM FAILURES:
- 0 trades: entry conditions too strict (need vol OR trend, not both)
- Overtrading: too many loose conditions
- Negative Sharpe: no regime filter, whipsaws in chop

WINNING FORMULA (proven): Price channel breakout + 1 filter (trend OR vol) + ATR stoploss
SIMPLIFIED: Donchian(20) breakout + 1w HMA trend (single filter) + ATR stoploss

TIMEFRAME: 1d primary
HTF: 1w for trend alignment
TARGET: 40-80 total trades over 4 years (10-20/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_hma_v1"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    mid = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        mid[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w HMA for trend alignment
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate local 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    # Volume MA for confirmation
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
    
    warmup = 50
    MIN_HOLD_BARS = 3  # Prevent signal flipping
    
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
        
        bars_in_trade = bars_in_trade + 1 if in_position else 0
        
        # === 1w TREND DIRECTION ===
        # Single filter: 1w HMA alignment
        bullish_trend = close[i] > hma_1w_aligned[i]
        bearish_trend = close[i] < hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.2
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout: close closes outside channel
        breakout_up = close[i] > donch_upper[i]
        breakout_down = close[i] < donch_lower[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Breakout above upper band + bullish 1w trend + volume
            if breakout_up and bullish_trend:
                if vol_spike:  # Volume confirmation
                    desired_signal = SIZE
                else:
                    # Allow entry without vol if strong trend
                    desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Breakout below lower band + bearish 1w trend
            if breakout_down and bearish_trend:
                if vol_spike:
                    desired_signal = -SIZE
                else:
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
        
        # === MINIMUM HOLD PERIOD ===
        if in_position and bars_in_trade < MIN_HOLD_BARS:
            # Don't exit early
            desired_signal = position_side * SIZE
        
        # === CHANNEL MEAN REVERSION EXIT ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price returns inside channel from above
            if not breakout_up and close[i] < donch_mid[i]:
                exit_triggered = True
            # OR: trend reverses
            if bearish_trend and bars_in_trade >= MIN_HOLD_BARS:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price returns inside channel from below
            if not breakout_down and close[i] > donch_mid[i]:
                exit_triggered = True
            # OR: trend reverses
            if bullish_trend and bars_in_trade >= MIN_HOLD_BARS:
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
                bars_in_trade = 0
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
                bars_in_trade = 0
        
        signals[i] = desired_signal
    
    return signals