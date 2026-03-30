#!/usr/bin/env python3
"""
Experiment #028: 1d RSI Mean-Reversion + 1w KAMA Trend + Volume Confirmation

HYPOTHESIS: RSI extremes (oversold/overbought) combined with multi-timeframe
KAMA trend alignment captures high-probability mean-reversion trades while
avoiding counter-trend trades. 1d timeframe naturally limits trades to ~1-2/month.

WHY IT WORKS IN BOTH BULL AND BEAR:
- In bull: RSI<30 bounces are quick reversals up (fear->greed)
- In bear: RSI>70 shorts catch the continued downtrend
- 1w KAMA filters direction, avoiding counter-trend trades
- Volume confirms institutional reversals

TARGET: 50-100 total over 4 years (12-25/year). HARD MAX: 150.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_rsi_kama_1w_v1"
timeframe = "1d"
leverage = 1.0

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

def calculate_kama(close, period=30, fast_ema=2, slow_ema=30):
    """Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    direction = np.abs(close[period:] - close[:-period])
    volatility = np.zeros(n - period)
    for i in range(n - period):
        for j in range(period):
            volatility[i] += abs(close[i+j+1] - close[i+j])
    
    er = np.zeros(n)
    er[period:] = direction / np.where(volatility > 0, volatility, 1)
    
    # Fast and slow EMA constants
    fast_const = 2 / (fast_ema + 1)
    slow_const = 2 / (slow_ema + 1)
    sc = (er * (fast_const - slow_const) + slow_const) ** 2
    
    kama = np.full(n, np.nan)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1w HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w KAMA for trend direction (aligned to 1d)
    kama_1w = calculate_kama(df_1w['close'].values, period=20)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Local 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = pd.Series(close).rolling(window=14, min_periods=14).apply(
        lambda x: 100 - (100 / (1 + (x.diff().where(x.diff() > 0, 0)).rolling(14).mean() / 
                               (-(x.diff().where(x.diff() < 0, 0)).rolling(14).mean())).clip(0.0001)), raw=True
    ).fillna(50).values
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # KAMA for local trend (21 period)
    kama_local = calculate_kama(close, period=21)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(kama_local[i]):
            signals[i] = 0.0
            continue
        
        # === 1w TREND DIRECTION ===
        price_above_1w_kama = close[i] > kama_1w_aligned[i]
        kama_1w_rising = kama_1w_aligned[i] > kama_1w_aligned[i-1] if i > 0 else False
        is_bullish_1w = price_above_1w_kama and kama_1w_rising
        is_bearish_1w = not price_above_1w_kama and not kama_1w_rising
        
        # === LOCAL TREND (21 KAMA) ===
        price_above_local_kama = close[i] > kama_local[i]
        
        # === RSI EXTREMES (not mild readings) ===
        rsi_oversold = rsi_14[i] < 35  # True oversold
        rsi_overbought = rsi_14[i] > 65  # True overbought
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: RSI oversold + bullish 1w trend + volume
            if rsi_oversold and is_bullish_1w and price_above_local_kama:
                if vol_spike:
                    desired_signal = SIZE
                else:
                    # Still enter if strong local trend
                    desired_signal = SIZE * 0.5  # Half size without volume
            
            # === SHORT: RSI overbought + bearish 1w trend + volume
            if rsi_overbought and is_bearish_1w and not price_above_local_kama:
                if vol_spike:
                    desired_signal = -SIZE
                else:
                    desired_signal = -SIZE * 0.5  # Half size without volume
        
        # === STOPLOSS CHECK (2.5 ATR from entry) ===
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
        
        # === HOLDING PERIOD (min 5 days = 5 bars) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 5:
            # === TARGET EXIT: RSI normalized (40-60 range) ===
            if position_side > 0 and rsi_14[i] > 55:
                desired_signal = 0.0
            if position_side < 0 and rsi_14[i] < 45:
                desired_signal = 0.0
            
            # === TRAILING EXIT: Price crosses local KAMA ===
            if position_side > 0 and close[i] < kama_local[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > kama_local[i]:
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