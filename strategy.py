#!/usr/bin/env python3
"""
Experiment #021: 12h KAMA Trend + RSI Momentum + Volume

HYPOTHESIS: KAMA(10) tracks price efficiently in both bull and bear markets.
RSI(14) crossing 50 is a proven momentum shift signal. Volume confirmation 
filters false breakouts. 1d HMA provides trend bias. This combination 
captures major trend changes with 2-3 conditions only.

TIMEFRAME: 12h
HTF: 1d for trend bias (HMA21)
TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close[period:] - close[:-period])
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - close[i-period:i]))
    
    er = np.zeros(n)
    mask = volatility > 1e-10
    er[mask] = change[mask] / volatility[mask]
    
    # Calculate smoothing constant
    fast_const = 2 / (fast + 1)
    slow_const = 2 / (slow + 1)
    sc = (er * (fast_const - slow_const) + slow_const) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan, dtype=np.float64)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
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
    """RSI calculation"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return (100 - (100 / (1 + rs))).values

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
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_10 = calculate_kama(close, period=10)
    rsi_14 = calculate_rsi(close, period=14)
    rsi_prev = np.roll(rsi_14, 1)
    rsi_prev[0] = np.nan
    
    # Volume MA
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
        
        if np.isnan(kama_10[i]) or np.isnan(rsi_14[i]):
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
        
        # === TREND BIAS (1d HMA) ===
        bullish_trend = close[i] > hma_1d_aligned[i]
        bearish_trend = close[i] < hma_1d_aligned[i]
        
        # === MOMENTUM (RSI crossover) ===
        rsi_curr = rsi_14[i]
        rsi_prv = rsi_prev[i] if i > 0 else np.nan
        rsi_cross_up = (rsi_curr > 50) and (rsi_prv <= 50) if not np.isnan(rsi_prv) else False
        rsi_cross_down = (rsi_curr < 50) and (rsi_prv >= 50) if not np.isnan(rsi_prv) else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.2
        
        # === KAMA SLOPE for trend confirmation ===
        kama_up = kama_10[i] > kama_10[i-1] if i > 0 and not np.isnan(kama_10[i-1]) else False
        kama_down = kama_10[i] < kama_10[i-1] if i > 0 and not np.isnan(kama_10[i-1]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # RSI crosses above 50 + bullish 1d trend + volume confirmation
            if rsi_cross_up and bullish_trend and vol_confirm:
                desired_signal = SIZE
            # Alternative: Strong momentum (RSI > 60) + KAMA rising + trend
            elif rsi_curr > 60 and kama_up and bullish_trend and vol_confirm:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # RSI crosses below 50 + bearish 1d trend + volume confirmation
            if rsi_cross_down and bearish_trend and vol_confirm:
                desired_signal = -SIZE
            # Alternative: Strong bearish momentum (RSI < 40) + KAMA falling + trend
            elif rsi_curr < 40 and kama_down and bearish_trend and vol_confirm:
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
        
        # === EXIT LOGIC ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: RSI overbought OR trend reversal
            if rsi_curr > 75:
                exit_triggered = True
            if bearish_trend and rsi_curr < 45:
                exit_triggered = True
            # Stop on trend reversal
            if close[i] < hma_1d_aligned[i] and rsi_curr < 50:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: RSI oversold OR trend reversal
            if rsi_curr < 25:
                exit_triggered = True
            if bullish_trend and rsi_curr > 55:
                exit_triggered = True
            # Stop on trend reversal
            if close[i] > hma_1d_aligned[i] and rsi_curr > 50:
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
        
        signals[i] = desired_signal
    
    return signals