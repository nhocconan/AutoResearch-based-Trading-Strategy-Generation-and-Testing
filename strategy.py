#!/usr/bin/env python3
"""
Experiment #011: 6h Williams Range Expansion + Volume + 1d Trend

HYPOTHESIS: Williams Range Expansion triggers on momentum acceleration after
a pullback. Price opens near the prior range extreme, then continues - this
captures institutional moves. Combined with 1d HMA trend alignment and volume
confirmation, this filters false breakouts. 6h is slower than 4h, reducing
trade frequency. Works in both bull (long breakouts above 1d HMA) and bear
(short breakdowns below 1d HMA) markets.

TIMEFRAME: 6h primary
HTF: 1d for trend bias
TARGET: 50-150 total trades over 4 years (12-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williams_range_exp_1d_v1"
timeframe = "6h"
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

def calculate_williams_range(high, low, close, period=2):
    """Williams Range Expansion: open position relative to prior range"""
    n = len(high)
    if n < period + 2:
        return np.full(n, np.nan)
    
    # Range of prior bar
    prior_range = high[1:] - low[1:]
    prior_range = np.concatenate([[np.nan], prior_range])
    
    # Open position relative to prior range
    # positive = opened near high, negative = opened near low
    open_pos = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if prior_range[i] > 0:
            open_pos[i] = (close[i-1] - low[i]) / prior_range[i]  # 0=low, 1=high
    
    return open_pos, prior_range

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
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
    open_prices = prices["open"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Williams Range Expansion (K=0.5)
    K = 0.5
    open_pos, prior_range = calculate_williams_range(high, low, close, period=2)
    
    # Donchian for structure
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for momentum
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    # ADX for trend strength
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr_smooth = atr_14.copy()
    plus_dm_ema = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_ema = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    plus_di = 100 * plus_dm_ema / (atr_smooth + 1e-10)
    minus_di = 100 * minus_dm_ema / (atr_smooth + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, min_periods=28, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.25  # Moderate size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === ADX REGIME ===
        adx_val = adx[i] if not np.isnan(adx[i]) else 25
        trending = adx_val > 22
        
        # === WILLIAMS RANGE EXPANSION ===
        op = open_pos[i] if not np.isnan(open_pos[i]) else 0.5
        pr = prior_range[i] if not np.isnan(prior_range[i]) else atr_14[i]
        
        # Long setup: opened near low (op < 0.4), price rallies
        williams_long = op < 0.4 and close[i] > open_prices[i]
        # Short setup: opened near high (op > 0.6), price drops
        williams_short = op > 0.6 and close[i] < open_prices[i]
        
        # === DONCHIAN BREAKOUT ===
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Williams range expansion UP + volume + trend aligned + trending
            if williams_long and vol_spike and price_above_1d_hma and trending:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Williams range expansion DOWN + volume + against trend + trending
            if williams_short and vol_spike and not price_above_1d_hma and trending:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR) ===
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
        
        # === EXIT ON OPPOSITE SIGNAL OR TIME ===
        exit_triggered = False
        
        if in_position:
            bars_since_entry = i - entry_bar
            
            # Opposite channel breakout
            if position_side > 0 and price_below_lower:
                exit_triggered = True
            if position_side < 0 and price_above_upper:
                exit_triggered = True
            
            # RSI extreme
            if position_side > 0 and rsi[i] < 30:
                exit_triggered = True
            if position_side < 0 and rsi[i] > 70:
                exit_triggered = True
            
            # Time-based exit (max 20 bars per position to avoid holding)
            if bars_since_entry >= 20:
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
                bars_since_entry = 0
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
            else:
                # Same direction - maintain
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
                bars_since_entry = 0
        
        signals[i] = desired_signal
    
    return signals