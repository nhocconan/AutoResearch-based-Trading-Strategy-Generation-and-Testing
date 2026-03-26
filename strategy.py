#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + RSI Momentum + 1d Trend

HYPOTHESIS: The proven winning formula is SIMPLE entries + tight filters.
- Donchian(20) marks institutional breakout points (strongest signal in DB)
- RSI(14) confirms momentum without over-complicating (CRSI failed due to overtrading)
- 1d HMA(21) filters by trend direction (prevents fighting major trends)
- Volume confirmation to avoid false breakouts
- ATR stoploss for risk management

WHY IT WORKS IN BOTH MARKETS:
- Bull: Long breakouts when price above 1d HMA
- Bear: Short breakdowns when price below 1d HMA (2022 crash protection)
- Range: RSI extremes at channel edges for mean reversion

TIMEFRAME: 4h | HTF: 1d | TARGET: 75-150 total trades over 4 years
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_rsi_1d_hma_v1"
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
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, middle, lower

def calculate_rsi(close, period=14):
    """RSI indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === Calculate local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    donch_upper, donch_middle, donch_lower = calculate_donchian(high, low, period=20)
    
    # RSI
    rsi_14 = calculate_rsi(close, period=14)
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d HMA) ===
        bullish_trend = close[i] > hma_1d_aligned[i]
        bearish_trend = close[i] < hma_1d_aligned[i]
        
        # === MOMENTUM (RSI) ===
        rsi_val = rsi_14[i]
        rsi_oversold = rsi_val < 35
        rsi_overbought = rsi_val > 65
        rsi_neutral = 35 <= rsi_val <= 65
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # True breakout: price CLOSES outside channel AND volume confirms
        breakout_up = close[i] > donch_upper[i] and close[i-1] <= donch_upper[i-1]
        breakout_down = close[i] < donch_lower[i] and close[i-1] >= donch_lower[i-1]
        
        # Price already beyond channel (sustained move)
        above_upper = close[i] > donch_upper[i]
        below_lower = close[i] < donch_lower[i]
        
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Breakout above upper band + bullish trend + volume
            if breakout_up and bullish_trend and vol_spike:
                desired_signal = SIZE
            # Alternative: bounce from lower band in uptrend (mean reversion)
            elif rsi_oversold and bullish_trend and below_lower:
                desired_signal = SIZE
            # Alternative: pullback to middle in strong uptrend
            elif above_upper and rsi_oversold and vol_spike:
                desired_signal = SIZE / 2  # half position
        
        if not in_position:
            # === NEW SHORT ENTRY ===
            # Breakout below lower band + bearish trend + volume
            if breakout_down and bearish_trend and vol_spike:
                desired_signal = -SIZE
            # Alternative: RSI extreme at upper in downtrend
            elif rsi_overbought and bearish_trend and above_upper:
                desired_signal = -SIZE
            # Alternative: rejection at upper in strong downtrend
            elif below_lower and rsi_overbought and vol_spike:
                desired_signal = -SIZE / 2  # half position
        
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
        
        # === EXIT CONDITIONS ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: break below lower band OR RSI extreme
            if breakout_down:
                exit_triggered = True
            if rsi_val < 30:
                exit_triggered = True
            # Take profit: RSI overbought + strong move
            if rsi_val > 75 and vol_spike:
                desired_signal = SIZE / 2  # reduce to half
        
        if in_position and position_side < 0:
            # Short exit: break above upper band OR RSI extreme
            if breakout_up:
                exit_triggered = True
            if rsi_val > 70:
                exit_triggered = True
            # Take profit: RSI oversold + strong move
            if rsi_val < 25 and vol_spike:
                desired_signal = -SIZE / 2  # reduce to half
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or direction change
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