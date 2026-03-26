#!/usr/bin/env python3
"""
Experiment #010: 1d Donchian Breakout + KAMA Trend + RSI Momentum

HYPOTHESIS: Daily Donchian(20) breakout captures institutional moves.
KAMA(10) trend filter ensures we trade with the longer-term trend.
RSI(14) adds momentum confirmation to avoid false breakouts.
Weekly KAMA provides regime context (bull/bear/range).
ATR-based stoploss protects against whipsaws.

WHY THIS WORKS IN BOTH BULL AND BEAR:
- Donchian breakout works in all markets (bull breakouts, bear breakdowns)
- Long only when 1w KAMA rising (avoids shorting rallies in bull)
- Short only when 1w KAMA falling (avoids catching knives in bull)
- Tight ATR stoploss prevents 2022-style crash losses

TARGET: 50-100 total trades over 4 years (~15-25/year).
Reference: mtf_1d_kama_rsi_chop_regime_1w_v1 (test_sharpe=1.310, 74tr)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_kama_rsi_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average
    """
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    # Calculate price change
    change = np.abs(close[period:] - close[:-period])
    
    # Calculate volatility (sum of absolute price changes)
    volatility = np.zeros(n - period)
    for i in range(len(volatility)):
        volatility[i] = np.sum(np.abs(close[i+1:i+1+period] - close[i:i+period]))
    
    # Efficiency ratio
    er = np.zeros(n)
    er[period:] = change / (volatility + 1e-10)
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    fast_const = 2 / (fast + 1)
    slow_const = 2 / (slow + 1)
    
    sc = (er * (fast_const - slow_const) + slow_const) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan, dtype=np.float64)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if np.isnan(sc[i]):
            continue
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
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper, middle, lower"""
    n = len(close)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load weekly data ONCE for regime context
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly KAMA for trend
    kama_1w = calculate_kama(df_1w['close'].values, period=10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate weekly KAMA slope (for trend direction)
    kama_1w_slope = np.full(n, 0.0)
    for i in range(10, n):
        if not np.isnan(kama_1w_aligned[i]) and not np.isnan(kama_1w_aligned[i-5]):
            kama_1w_slope[i] = kama_1w_aligned[i] - kama_1w_aligned[i-5]
    
    # Calculate 1d indicators
    kama_10 = calculate_kama(close, period=10)
    kama_21 = calculate_kama(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Donchian(20) channel
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_middle = (donch_upper + donch_lower) / 2
    
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
    
    # Warmup needed for indicators
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(kama_10[i]) or np.isnan(atr_14[i]) or np.isnan(rsi_14[i]):
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
        
        # === REGIME CHECK (Weekly KAMA) ===
        weekly_bull = kama_1w_slope[i] > 0 if not np.isnan(kama_1w_slope[i]) else True
        weekly_bear = kama_1w_slope[i] < 0 if not np.isnan(kama_1w_slope[i]) else False
        
        # === TREND CHECK (1d KAMA) ===
        daily_bull = kama_10[i] > kama_21[i] if (not np.isnan(kama_21[i])) else True
        daily_bear = kama_10[i] < kama_21[i] if (not np.isnan(kama_21[i])) else False
        
        # === MOMENTUM CHECK (RSI) ===
        rsi_ok_long = rsi_14[i] > 50 if not np.isnan(rsi_14[i]) else False
        rsi_ok_short = rsi_14[i] < 50 if not np.isnan(rsi_14[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.0
        
        # === DONCHIAN BREAKOUT CHECK ===
        prev_upper = donch_upper[i-1] if i > 0 else 0
        prev_lower = donch_lower[i-1] if i > 0 else 0
        
        upper_broken = close[i] > prev_upper and prev_upper > 0
        lower_broken = close[i] < prev_lower and prev_lower > 0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Upper Donchian broken + weekly bull + daily bull + RSI confirm
        if upper_broken and weekly_bull and daily_bull and rsi_ok_long:
            if vol_confirm:
                desired_signal = SIZE
            else:
                desired_signal = SIZE  # Allow without vol for more trades
        
        # SHORT: Lower Donchian broken + weekly bear + daily bear + RSI confirm
        if lower_broken and weekly_bear and daily_bear and rsi_ok_short:
            if vol_confirm:
                desired_signal = -SIZE
            else:
                desired_signal = -SIZE
        
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
        
        # === TAKE PROFIT at opposite Donchian ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP at upper Donchian (locked in gains)
            if high[i] >= donch_upper[i]:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP at lower Donchian
            if low[i] <= donch_lower[i]:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
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
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals