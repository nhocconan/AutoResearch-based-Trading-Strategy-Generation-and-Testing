#!/usr/bin/env python3
"""
Experiment #007: 6h Bollinger Squeeze + ATR Volatility Expansion + Daily HMA Trend

HYPOTHESIS: Bollinger Band squeeze (volatility compression) followed by ATR 
breakout (volatility expansion) captures institutional accumulation/distribution 
cycles. Daily HMA trend filters ensure we trade WITH major trend, not against it.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Bull markets: Daily HMA up, buy squeeze breaks to upside
- Bear markets: Daily HMA down, sell squeeze breaks to downside  
- Volatility expansion after compression is symmetric — works in both directions
- ATR filter removes false breakouts in low-volatility choppy periods

KEY DESIGN (inspired by CRSI regime success pattern):
1. Daily HMA(21) for trend direction (filters both long/short)
2. 6h Bollinger Band squeeze detection (BW < 20-bar avg)
3. 6h ATR breakout confirmation (ATR > 20-bar avg) — confirms genuine move
4. Volume spike confirmation (1.5x 20-avg)
5. Price outside Bollinger bands confirms direction
6. Tight ATR-based stoploss

TARGET: 75-125 total trades over 4 years (19-31/year) — fits 6h HARD MAX of 300.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_squeeze_atr_vol_expansion_1d_v1"
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    
    return upper, mid, lower

def calculate_bollinger_width(close, period=20, std_dev=2.0):
    """Bollinger Band Width (for squeeze detection)"""
    upper, mid, lower = calculate_bollinger_bands(close, period, std_dev)
    n = len(close)
    width = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if mid[i] > 1e-10:
            width[i] = (upper[i] - lower[i]) / mid[i]
        else:
            width[i] = np.nan
    
    return width

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Price change ratio
    change = np.abs(close[period:] - close[:-period])
    
    # Volatility (sum of price changes)
    volatility = np.zeros(n - period, dtype=np.float64)
    for i in range(n - period):
        volatility[i] = np.sum(np.abs(close[i+1:i+period+1] - close[i:i+period]))
    
    # Efficiency ratio (ER)
    er = np.zeros(n, dtype=np.float64)
    er[period:] = np.where(volatility > 1e-10, change / volatility, 0.0)
    er = np.clip(er, 0.0, 1.0)
    
    # Smoothing constant
    fast_const = 2.0 / (fast + 1)
    slow_const = 2.0 / (slow + 1)
    sc = (er * (fast_const - slow_const) + slow_const) ** 2
    
    kama = np.full(n, np.nan, dtype=np.float64)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load daily data for trend
    df_1d = get_htf_data(prices, '1d')
    
    # Daily HMA(21) for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Daily ATR for regime filter
    atr_1d_raw = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_raw)
    
    # Daily KAMA for entry confirmation
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # === 6h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ATR average for breakout detection
    atr_avg = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # Bollinger Band width for squeeze detection
    bb_width = calculate_bollinger_width(close, period=20, std_dev=2.0)
    bb_width_avg = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    
    # Bollinger Bands for price levels
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # EMA for direction
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
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
    
    # Cooldown to prevent overtrading (wait at least 4 bars after exit)
    bars_since_exit = 999
    
    # Warmup - need enough bars for all indicators
    warmup = 100
    
    for i in range(warmup, n):
        bars_since_exit += 1
        
        # Check all required indicators
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            continue
        
        if np.isnan(bb_width[i]) or np.isnan(bb_width_avg[i]):
            signals[i] = 0.0
            in_position = False
            continue
        
        if np.isnan(atr_avg[i]) or atr_avg[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            continue
        
        if np.isnan(ema_21[i]):
            signals[i] = 0.0
            in_position = False
            continue
        
        # Daily trend (need at least 2 days of data)
        daily_trend_up = False
        daily_trend_down = False
        if not np.isnan(hma_1d_aligned[i]) and i > 10:
            # Compare current HMA to HMA 5 bars ago (approximately 5 days)
            if i >= 10 and not np.isnan(hma_1d_aligned[i-10]):
                if hma_1d_aligned[i] > hma_1d_aligned[i-10]:
                    daily_trend_up = True
                elif hma_1d_aligned[i] < hma_1d_aligned[i-10]:
                    daily_trend_down = True
        
        # === SQUEEZE DETECTION ===
        # Squeeze fires when BB width < 20-bar average (compression)
        squeeze_fired = bb_width[i] < bb_width_avg[i]
        
        # ATR breakout: current ATR > 20-bar average (expansion after compression)
        atr_breakout = atr_14[i] > atr_avg[i]
        
        # Price outside Bollinger Bands (breakout confirmation)
        price_above_bb = close[i] > bb_upper[i]
        price_below_bb = close[i] < bb_lower[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Entry conditions
        desired_signal = 0.0
        
        # LONG: Squeeze + ATR breakout + price above BB + bullish daily trend + volume
        if not in_position and bars_since_exit >= 4:
            if squeeze_fired and atr_breakout and price_above_bb and daily_trend_up and vol_spike:
                desired_signal = SIZE
            # Also allow entry without volume if very strong trend
            elif squeeze_fired and atr_breakout and price_above_bb and daily_trend_up:
                if close[i] > ema_21[i]:
                    desired_signal = SIZE * 0.5  # Half size without volume confirmation
        
        # SHORT: Squeeze + ATR breakout + price below BB + bearish daily trend + volume
        if not in_position and bars_since_exit >= 4:
            if squeeze_fired and atr_breakout and price_below_bb and daily_trend_down and vol_spike:
                desired_signal = -SIZE
            # Also allow entry without volume if very strong trend
            elif squeeze_fired and atr_breakout and price_below_bb and daily_trend_down:
                if close[i] < ema_21[i]:
                    desired_signal = -SIZE * 0.5  # Half size without volume confirmation
        
        # === STOPLOSS CHECK ===
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
            bars_since_exit = 0
        
        # === TAKE PROFIT at 3:1 R:R ===
        tp_triggered = False
        if in_position and position_side > 0:
            profit_target = entry_price + 3.0 * entry_atr
            if high[i] >= profit_target:
                tp_triggered = True
        
        if in_position and position_side < 0:
            profit_target = entry_price - 3.0 * entry_atr
            if low[i] <= profit_target:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
            bars_since_exit = 0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
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
                # Only exit if stoploss or TP hit, not just because signal is 0
                # (signal could be 0 because squeeze conditions not met but we're still in valid trend)
                pass
        
        signals[i] = desired_signal if in_position else 0.0
    
    return signals