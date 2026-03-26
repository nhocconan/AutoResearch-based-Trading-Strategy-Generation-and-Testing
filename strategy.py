#!/usr/bin/env python3
"""
Experiment #008: 12h Donchian Breakout with 1w Trend + Volume Confirmation

Hypothesis: Simple 12h Donchian(20) breakout with 1w SMA trend filter and 
volume confirmation should work because:
1. Donchian breakout = proven price structure (captures major trends)
2. 1w SMA = structural trend direction (filters noise on 12h)
3. Volume spike = confirms genuine breakout (eliminates false breaks)
4. 12h timeframe = 50-150 trades/4yr (manages fee drag)
5. 2x ATR stoploss = protects against crashes like 2022

Why simpler is better:
- 25 failed experiments used complex multi-condition logic
- DB winners (Sharpe 1.3-1.5) use simple price channel + volume + regime
- Fewer conditions = fewer trades = less fee drag = better generalization

Entry: 1w trend-aligned + Donchian break + volume spike
Target: Sharpe>0.5, trades≥50 train, trades≥10 test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1w_sma_volume_v1"
timeframe = "12h"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - price channel breakout system"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_volume_ma(volume, period=20):
    """Volume moving average for spike detection"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def calculate_sma(close, period):
    """Simple Moving Average"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w SMA and align to 12h
    sma_50_1w_raw = calculate_sma(df_1w['close'].values, 50)
    sma_50_1w = align_htf_to_ltf(prices, df_1w, sma_50_1w_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_in_position = 0
    
    # Warmup period
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
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
        
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_50_1w[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1w TREND FILTER ===
        price_above_1w = close[i] > sma_50_1w[i]
        price_below_1w = close[i] < sma_50_1w[i]
        
        # === VOLUME SPIKE CONFIRMATION ===
        vol_ratio = volume[i] / vol_ma[i]
        vol_spike = vol_ratio > 1.5
        
        # === DONCHIAN BREAKOUT ===
        # Must have CLOSE above/below previous upper/lower (closed candle confirms)
        donch_break_long = False
        donch_break_short = False
        
        if i > 0:
            if not np.isnan(donch_upper[i-1]) and close[i] > donch_upper[i-1]:
                donch_break_long = True
            if not np.isnan(donch_lower[i-1]) and close[i] < donch_lower[i-1]:
                donch_break_short = True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: 1w bullish + Donchian break + volume spike
            if price_above_1w and donch_break_long and vol_spike:
                desired_signal = SIZE
            
            # SHORT: 1w bearish + Donchian break + volume spike
            elif price_below_1w and donch_break_short and vol_spike:
                desired_signal = -SIZE
        
        else:
            # === TRAILING STOPLOSS (2x ATR) ===
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.0 * entry_atr
                
                # Stop out if price drops below trailing stop
                if low[i] < trailing_stop:
                    desired_signal = 0.0
                else:
                    desired_signal = SIZE
            
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.0 * entry_atr
                
                # Stop out if price rises above trailing stop
                if high[i] > trailing_stop:
                    desired_signal = 0.0
                else:
                    desired_signal = -SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            final_signal = SIZE
        elif desired_signal < 0:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or reversal
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                bars_in_position = 0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                bars_in_position = 0
        
        if in_position:
            bars_in_position += 1
        
        signals[i] = final_signal
    
    return signals