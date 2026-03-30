#!/usr/bin/env python3
"""
Experiment #004: 1d Donchian Breakout + Weekly Trend + Volume (BTC/ETH/SOL)

HYPOTHESIS: Simple daily Donchian(20) breakout with weekly trend filter
and volume confirmation. Works in both bull and bear:
- Bull: Price breaks above 20d high + weekly uptrend = long
- Bear: Price breaks below 20d low + weekly downtrend = short
- Range: No trades (price in channel + weak weekly = flat)

WHY IT SHOULD WORK: 
- Donchian breakout is a proven price structure edge
- Weekly filter prevents trading against major trend
- Volume confirms the breakout isn't false
- ATR stoploss manages risk in both directions
- Simple = few trades = low fee drag = better test generalization

TARGET: 50-100 total trades over 4 years (12-25/year). HARD MAX: 150.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_vol_v1"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel: upper = highest high, lower = lowest low"""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle

def calculate_sma(data, period):
    """Simple Moving Average with min_periods"""
    return pd.Series(data).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # === Weekly trend: 8-week SMA vs price ===
    weekly_close = df_1w['close'].values
    weekly_sma_8 = calculate_sma(weekly_close, 8)
    
    # Weekly trend: up if price > SMA(8), down if price < SMA(8)
    weekly_up = weekly_close > weekly_sma_8
    weekly_down = weekly_close < weekly_sma_8
    
    # Weekly ATR for sizing
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_atr = calculate_atr(weekly_high, weekly_low, weekly_close, period=14)
    
    # Align weekly to daily
    weekly_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_up.astype(float))
    weekly_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_down.astype(float))
    
    # === Daily indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    # Volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 60  # Donchian needs 20, ATR needs 14
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            continue
        
        # Weekly alignment check
        weekly_up = weekly_up_aligned[i] > 0.5 if not np.isnan(weekly_up_aligned[i]) else False
        weekly_down = weekly_down_aligned[i] > 0.5 if not np.isnan(weekly_down_aligned[i]) else False
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Donchian breakout signals
        # Long: price breaks above 20d high with volume
        bull_breakout = close[i] > donch_upper[i] and close[i-1] <= donch_upper[i-1]
        # Short: price breaks below 20d low with volume
        bear_breakout = close[i] < donch_lower[i] and close[i-1] >= donch_lower[i-1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + weekly uptrend + volume
            if bull_breakout and weekly_up and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Bearish breakout + weekly downtrend + volume
            elif bear_breakout and weekly_down and vol_spike:
                desired_signal = -SIZE
        
        # === TRAILING STOP (2.5 ATR) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: price falls 2.5 ATR from trailing high
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly turns bearish
                if weekly_down:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: price rises 2.5 ATR from trailing low
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly turns bullish
                if weekly_up:
                    desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals