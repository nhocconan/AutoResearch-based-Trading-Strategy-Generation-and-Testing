#!/usr/bin/env python3
"""
Experiment #006: 12h Donchian Breakout + Volume Spike + 1d EMA200

HYPOTHESIS: Price breaking above/below 20-bar Donchian channels captures momentum shifts.
Volume spike confirms institutional involvement. 1d EMA200 filters for trend direction.
12h timeframe = ~3x fewer trades than 4h = less fee drag.

WHY IT WORKS IN BULL AND BEAR:
- Bull: Long breakouts above EMA200 when price breaks 20h high + vol spike
- Bear: Short breakouts below EMA200 when price breaks 20h low + vol spike
- Range: Choppy, fewer breakouts, smaller losses

ENTRY CONDITIONS:
- LONG: price > 1d EMA200 AND close crosses ABOVE 12h Donchian high(20) AND vol_ratio > 1.5
- SHORT: price < 1d EMA200 AND close crosses BELOW 12h Donchian low(20) AND vol_ratio > 1.5

EXIT: 2.5 ATR stoploss, take profit at 2R, or EMA200 cross

TARGET: 60-120 total trades over 4 years = 15-30/year
Signal size: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_ema200_1d_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for trend (slower = fewer signals, more robust)
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 periods = 10 days on 12h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume ratio (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = max(220, donchian_period + 20)  # Need enough for EMA200 alignment
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if EMA not aligned
        if np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if Donchian not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA200) ===
        price_above_ema200 = close[i] > ema_200_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Previous bar values for comparison
        prev_close = close[i - 1] if i > 0 else close[0]
        prev_high = high[i - 1] if i > 0 else high[0]
        prev_low = low[i - 1] if i > 0 else low[0]
        prev_donchian_high = donchian_high[i - 1] if i > 0 else donchian_high[0]
        prev_donchian_low = donchian_low[i - 1] if i > 0 else donchian_low[0]
        
        # Breakout: price closes above/below previous Donchian high/low
        bullish_breakout = (prev_close <= prev_donchian_high) and (close[i] > donchian_high[i])
        bearish_breakout = (prev_close >= prev_donchian_low) and (close[i] < donchian_low[i])
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Above EMA200 + bullish breakout + volume spike
            if price_above_ema200 and bullish_breakout and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Below EMA200 + bearish breakout + volume spike
            if not price_above_ema200 and bearish_breakout and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position:
            # Update highest/lowest since entry
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            
            if position_side > 0:
                # Long: trailing stop rises with price
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                if low[i] < trailing_stop:
                    desired_signal = 0.0
                # Take profit at 2R
                profit_target = entry_price + 2.0 * entry_atr
                if close[i] >= profit_target:
                    desired_signal = SIZE / 2  # Half position
            
            if position_side < 0:
                # Short: trailing stop falls with price
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                if high[i] > trailing_stop:
                    desired_signal = 0.0
                # Take profit at 2R
                profit_target = entry_price - 2.0 * entry_atr
                if close[i] <= profit_target:
                    desired_signal = -SIZE / 2  # Half position
        
        # === EMA200 EXIT (trend reversal) ===
        if in_position:
            if position_side > 0 and not price_above_ema200:
                # Price crossed below EMA200, exit long
                desired_signal = 0.0
            if position_side < 0 and price_above_ema200:
                # Price crossed above EMA200, exit short
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals