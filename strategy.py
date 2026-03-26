#!/usr/bin/env python3
"""
Experiment #010: 1d KAMA + RSI + Volume with 1w Trend Filter

HYPOTHESIS: KAMA (adaptive moving average) captures trend shifts in both bull 
and bear markets. RSI at extremes (<35 long, >65 short) catches mean reversion.
1w KAMA filters direction. Volume spike confirms institutional moves.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- KAMA adapts to volatility - works in trending and choppy markets
- RSI extremes work in both directions for mean reversion
- 1w trend filter prevents fighting major trends

TARGET: 50-100 total trades over 4 years (proven from DB: mtf_1d_kama_rsi_chop_regime_1w_v1)
DB Reference: SOLUSDT test Sharpe=1.310, 74 trades, 46% win rate

KEY DESIGN:
1. 1w KAMA for trend direction (call once, align properly)
2. 1d RSI(14) for entry timing (<35 long, >65 short)
3. Volume spike confirmation (>1.3x 20-avg)
4. ATR-based stoploss (3x ATR)
5. Signal: 0.25 (discrete)
6. Cooldown: minimum 5 bars between entries
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi_vol_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=21, fast_ema=2, slow_ema=30):
    """
    Kaufman's Adaptive Moving Average
    ER = abs(change) / sum(abs(change)) over period
    smoothing = ER * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)
    """
    n = len(close)
    if n < period + slow_ema:
        return np.full(n, np.nan)
    
    # Calculate price changes
    change = np.abs(np.diff(close, prepend=close[0]))
    
    # Sum of absolute changes over period
    sum_change = np.zeros(n)
    for i in range(period - 1, n):
        sum_change[i] = np.sum(change[i - period + 1:i + 1])
    
    # Absolute price change over period
    abs_change = np.abs(close - np.roll(close, period - 1))
    abs_change[:period] = change[:period]
    
    # Efficiency Ratio
    er = np.where(sum_change > 1e-10, abs_change / sum_change, 0.5)
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    fast = 2 / (fast_ema + 1)
    slow = 2 / (slow_ema + 1)
    smoothing = er * (fast - slow) + slow
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        if np.isnan(kama[i - 1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + smoothing[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    deltas = np.diff(close, prepend=close[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.where(avg_loss > 1e-10, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
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
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w KAMA
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=21)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate 1d indicators
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Cooldown to prevent overtrading
    bars_since_entry = 999
    
    # Warmup for indicators
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1w_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1w TREND DIRECTION ===
        price_above_1w_kama = close[i] > kama_1w_aligned[i]
        price_below_1w_kama = close[i] < kama_1w_aligned[i]
        
        # === RSI ENTRY CONDITIONS ===
        rsi = rsi_14[i]
        rsi_oversold = rsi < 35
        rsi_overbought = rsi > 65
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: RSI oversold + price above 1w KAMA + volume confirmation
        if rsi_oversold and price_above_1w_kama and vol_spike:
            desired_signal = SIZE
        # LONG weaker: RSI oversold + strong trend (no volume needed)
        elif rsi < 30 and price_above_1w_kama:
            desired_signal = SIZE
        
        # SHORT: RSI overbought + price below 1w KAMA + volume confirmation
        if rsi_overbought and price_below_1w_kama and vol_spike:
            desired_signal = -SIZE
        # SHORT weaker: RSI overbought + strong downtrend
        elif rsi > 70 and price_below_1w_kama:
            desired_signal = -SIZE
        
        # === COOLDOWN: minimum 5 bars between entries ===
        bars_since_entry += 1
        if desired_signal != 0.0:
            if in_position and bars_since_entry < 5:
                desired_signal = 0.0  # Force flat during cooldown
        
        # === STOPLOSS CHECK ===
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
            bars_since_entry = 0  # Reset cooldown after stoploss
        
        # === TRAILING STOP PROTECTION ===
        if in_position and not stoploss_triggered:
            if position_side > 0:
                # Trail stop on longs: move up as price rises
                profit = (close[i] - entry_price) / entry_atr
                if profit > 2.0:  # 2R profit, tighten stop
                    new_stop = close[i] - 2.0 * entry_atr
                    stop_price = max(stop_price, new_stop)
            elif position_side < 0:
                # Trail stop on shorts
                profit = (entry_price - close[i]) / entry_atr
                if profit > 2.0:
                    new_stop = close[i] + 2.0 * entry_atr
                    stop_price = min(stop_price, new_stop)
        
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
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
                bars_since_entry = 0
        elif desired_signal == 0.0 and in_position:
            # Exit signal but not stopped out - check if still in valid position
            if position_side > 0 and rsi_overbought:
                desired_signal = 0.0  # Exit long on RSI overbought
            elif position_side < 0 and rsi_oversold:
                desired_signal = 0.0  # Exit short on RSI oversold
        
        if desired_signal == 0.0:
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