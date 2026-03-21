#!/usr/bin/env python3
"""
Experiment #033: 1h Donchian Breakout + 4h HMA Trend + Volume Confirmation
Hypothesis: 1h timeframe captures multi-day swings better than intraday noise.
Donchian(20) breakouts catch momentum moves when price breaks 20-bar high/low.
4h HMA(21) provides major trend filter - only trade breakouts in trend direction.
Volume confirmation filters false breakouts (real moves have volume).
RSI(14) momentum filter ensures we're not entering at exhaustion.
Multiple entry triggers ensure ≥10 trades while 4h filter prevents counter-trend disasters.
Position sizing 0.30 with 2.5x ATR stoploss protects against 2022-style crashes.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_4h_hma_vol_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-bar high/low)."""
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    return donchian_high, donchian_low, donchian_mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, 20)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.nanmean(volume))
    
    # 1h HMA for additional trend confirmation
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # 4h trend filter (major regime)
        trend_4h_bullish = hma_4h_aligned[i] > 0 and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_aligned[i] > 0 and close[i] < hma_4h_aligned[i]
        
        # 1h HMA trend confirmation
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_high[i-1] if not np.isnan(donchian_high[i-1]) else False
        breakout_short = close[i] < donchian_low[i-1] if not np.isnan(donchian_low[i-1]) else False
        
        # Previous bar was inside channel (confirms breakout)
        prev_inside_long = close[i-1] <= donchian_high[i-1] if not np.isnan(donchian_high[i-1]) else False
        prev_inside_short = close[i-1] >= donchian_low[i-1] if not np.isnan(donchian_low[i-1]) else False
        
        # Volume confirmation (1.5x average for strong breakout)
        vol_confirm = volume[i] > vol_sma[i] * 1.3 if vol_sma[i] > 0 else True
        
        # RSI momentum (not overextended)
        rsi_bullish = rsi[i] > 45 and rsi[i] < 75
        rsi_bearish = rsi[i] > 25 and rsi[i] < 55
        rsi_rising = rsi[i] > rsi[i-2] if i > 2 else True
        rsi_falling = rsi[i] < rsi[i-2] if i > 2 else True
        
        # Entry logic - MULTIPLE triggers to ensure trades (Rule 9)
        new_signal = 0.0
        
        # LONG ENTRY TRIGGERS
        # Trigger 1: Donchian breakout + 4h bullish trend + volume
        if breakout_long and prev_inside_long and trend_4h_bullish and vol_confirm:
            new_signal = SIZE
        # Trigger 2: Donchian breakout + 1h HMA trend + RSI ok
        elif breakout_long and prev_inside_long and hma_trend_long and rsi_bullish:
            new_signal = SIZE
        # Trigger 3: 4h bullish + 1h HMA bull + RSI rising (trend continuation)
        elif trend_4h_bullish and hma_trend_long and rsi_rising and rsi[i] > 50:
            new_signal = SIZE
        # Trigger 4: Price above Donchian mid + 4h trend + volume
        elif close[i] > donchian_mid[i] and trend_4h_bullish and vol_confirm:
            new_signal = SIZE
        
        # SHORT ENTRY TRIGGERS
        # Trigger 1: Donchian breakout + 4h bearish trend + volume
        if breakout_short and prev_inside_short and trend_4h_bearish and vol_confirm:
            new_signal = -SIZE
        # Trigger 2: Donchian breakout + 1h HMA trend + RSI ok
        elif breakout_short and prev_inside_short and hma_trend_short and rsi_bearish:
            new_signal = -SIZE
        # Trigger 3: 4h bearish + 1h HMA bear + RSI falling (trend continuation)
        elif trend_4h_bearish and hma_trend_short and rsi_falling and rsi[i] < 50:
            new_signal = -SIZE
        # Trigger 4: Price below Donchian mid + 4h trend + volume
        elif close[i] < donchian_mid[i] and trend_4h_bearish and vol_confirm:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based with trailing
        if position_side > 0 and entry_price > 0:
            # Update highest since entry for trailing
            if close[i] > highest_since_entry:
                highest_since_entry = close[i]
            
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop: 2.5 ATR below highest
                new_trailing = highest_since_entry - 2.5 * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if close[i] < trailing_stop and trailing_stop > 0:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] > entry_price + 3.0 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            # Update lowest since entry for trailing
            if close[i] < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = close[i]
            
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop: 2.5 ATR above lowest
                new_trailing = lowest_since_entry + 2.5 * atr[i]
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                if close[i] > trailing_stop and trailing_stop > 0:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] < entry_price - 3.0 * atr[i] and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_since_entry = close[i] if position_side > 0 else 0.0
            lowest_since_entry = close[i] if position_side < 0 else 0.0
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals