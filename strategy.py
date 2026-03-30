#!/usr/bin/env python3
"""
Experiment #007: 6h RSI Pullback + Weekly EMA200 Trend + Volume

HYPOTHESIS: In strong uptrends (price > weekly EMA200), RSI < 35 signals mean
reversion pullback entries with excellent risk/reward because you're buying dips
within confirmed uptrends. Same logic for shorts in downtrends.

WHY NOVEL: Uses 1w EMA200 as regime filter + 6h RSI(14) for entry timing + 
volume confirmation. Not a breakout - this is PULLBACK within trend.

WHY 6h: Slower than 4h = fewer signals = less fee drag. Pullback strategies
need slower TF to avoid getting whipsawed by noise.

TARGET: 75-150 total trades over 4 years = 19-37/year.
Signal size: 0.25.

Entry: RSI < 35 + vol spike + price above weekly EMA200 = LONG pullback
Entry: RSI > 65 + vol spike + price below weekly EMA200 = SHORT rally

Exit: Trailing stop 2.5 ATR or RSI mean reversion (RSI > 55 for longs)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_rsi_pullback_wema200_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_rsi(prices_arr, period=14):
    """Calculate RSI using numpy"""
    n = len(prices_arr)
    if n < period + 1:
        return np.full(n, np.nan)
    
    deltas = np.diff(prices_arr, prepend=prices_arr[0])
    deltas[0] = 0
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period-1] = np.mean(gains[1:period])
    avg_loss[period-1] = np.mean(losses[1:period])
    
    for i in range(period, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

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
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA200 for trend direction (very slow, institutional)
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 6h indicators ===
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (confirms momentum)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
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
    
    warmup = 500  # Need 200 for weekly EMA200 alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if weekly EMA not aligned
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1w EMA200) ===
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === RSI PULLBACK ENTRY CONDITIONS ===
        # Long: RSI oversold (<35) in uptrend = buy the dip
        # Short: RSI overbought (>65) in downtrend = fade the rally
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: RSI oversold + vol spike + uptrend ===
            if uptrend and rsi_oversold and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: RSI overbought + vol spike + downtrend ===
            if downtrend and rsi_overbought and vol_spike:
                desired_signal = -SIZE
        
        # === TRAILING STOPLOSS (2.5 ATR) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === RSI MEAN REVERSION EXIT ===
        # Exit longs when RSI normalizes (>55)
        # Exit shorts when RSI normalizes (<45)
        if in_position and position_side > 0:
            if rsi_14[i] > 55:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if rsi_14[i] < 45:
                desired_signal = 0.0
        
        # === MINIMUM HOLDING PERIOD (6 bars = 1.5 days) ===
        # Prevents being stopped out by noise
        bars_held = i - entry_bar
        if in_position and bars_held < 6:
            # Don't exit early due to RSI normalization
            if position_side > 0 and rsi_14[i] > 55:
                if desired_signal == 0.0:
                    desired_signal = SIZE  # Keep holding
            if position_side < 0 and rsi_14[i] < 45:
                if desired_signal == 0.0:
                    desired_signal = -SIZE  # Keep holding
        
        # === MAX HOLDING PERIOD (24 bars = 6 days) ===
        # Force exit to avoid holding through range
        if in_position and bars_held >= 24:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
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
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals