#!/usr/bin/env python3
"""
Experiment #260: 30m Donchian Breakout with 4h HMA Trend Filter
Hypothesis: Simple Donchian channel breakouts on 30m with 4h HMA trend filter
can capture momentum moves while avoiding whipsaws. Fewer filters = more trades.
Key insight from 259 failed experiments: TOO MANY CONDITIONS = 0 TRADES.
This strategy uses MINIMAL filters: just trend direction + breakout + volume confirmation.
Position sizing: 0.25 entry, 0.125 half at 2R. Stoploss: 2.5*ATR trailing.
Target: Generate 50+ trades per symbol, Sharpe > 0.5, DD < -30%.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_donchian_breakout_4h_hma_volume_atr_v1"
timeframe = "30m"
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
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (0-1, >0.5 = bullish)."""
    ratio = np.where(volume > 0, taker_buy_volume / volume, 0.5)
    return ratio

def calculate_roc(close, period=10):
    """Calculate Rate of Change for momentum confirmation."""
    roc = np.zeros(len(close))
    for i in range(period, len(close)):
        roc[i] = (close[i] - close[i - period]) / close[i - period] * 100
    return roc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    roc = calculate_roc(close, 10)
    
    # Track previous values for breakout detection
    prev_donchian_upper = np.roll(donchian_upper, 1)
    prev_donchian_lower = np.roll(donchian_lower, 1)
    prev_donchian_upper[0] = donchian_upper[0]
    prev_donchian_lower[0] = donchian_lower[0]
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filter (4h HMA - simple and effective)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # Donchian breakout detection
        breakout_long = close[i] > prev_donchian_upper[i]
        breakout_short = close[i] < prev_donchian_lower[i]
        
        # Volume confirmation (loose threshold to ensure trades)
        vol_bullish = vol_ratio[i] > 0.48
        vol_bearish = vol_ratio[i] < 0.52
        
        # Momentum confirmation (loose threshold)
        mom_bullish = roc[i] > -2.0
        mom_bearish = roc[i] < 2.0
        
        # Price position in channel
        above_mid = close[i] > donchian_mid[i]
        below_mid = close[i] < donchian_mid[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Breakout long with trend confirmation (MINIMAL filters)
        if breakout_long:
            if trend_bullish:
                new_signal = SIZE_ENTRY
            elif above_mid and vol_bullish:
                new_signal = SIZE_ENTRY
        
        # Continuation long in uptrend
        elif trend_bullish and above_mid:
            if vol_bullish and mom_bullish:
                if close[i] > prev_close[i] * 1.002:  # price moving up
                    new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Breakout short with trend confirmation (MINIMAL filters)
        if breakout_short:
            if trend_bearish:
                new_signal = -SIZE_ENTRY
            elif below_mid and vol_bearish:
                new_signal = -SIZE_ENTRY
        
        # Continuation short in downtrend
        elif trend_bearish and below_mid:
            if vol_bearish and mom_bearish:
                if close[i] < prev_close[i] * 0.998:  # price moving down
                    new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals