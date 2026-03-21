#!/usr/bin/env python3
"""
Experiment #291: 1h Supertrend with 4h HMA Trend Filter + RSI Pullback
Hypothesis: 1h Supertrend captures momentum moves with better timing than 12h Donchian.
4h HMA provides trend bias (only trade Supertrend signals in trend direction).
RSI pullback filter ensures we enter on dips in uptrend (not chasing tops).
Volume confirmation filters false breakouts. ATR stoploss at 2.5*ATR.
Position sizing: 0.28 entry, 0.14 half at 2R profit. Discrete levels to minimize fees.
Target: Beat Sharpe=0.499 from current best (mtf_12h_supertrend_daily_hma_rsi_pullback_v2)
Why 1h: Faster reaction than 12h, but still enough bars to generate trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_supertrend_4h_hma_rsi_volume_atr_v1"
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
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    for i in range(1, len(close)):
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    return supertrend, direction

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

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Track previous values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_st_dir = np.roll(st_direction, 1)
    prev_st_dir[0] = st_direction[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # HTF trend filters
        fourh_bullish = close[i] > hma_4h_aligned[i]
        fourh_bearish = close[i] < hma_4h_aligned[i]
        
        # Supertrend signals
        st_long = st_direction[i] == 1 and prev_st_dir[i] == -1  # ST flip to bullish
        st_short = st_direction[i] == -1 and prev_st_dir[i] == 1  # ST flip to bearish
        
        # RSI filter (not too extreme, allow pullbacks)
        rsi_ok_long = 35 < rsi[i] < 70
        rsi_ok_short = 30 < rsi[i] < 65
        rsi_pullback_long = 40 < rsi[i] < 55  # Pullback in uptrend
        rsi_pullback_short = 45 < rsi[i] < 60  # Pullback in downtrend
        
        # Volume confirmation
        volume_ok = volume[i] > 1.2 * vol_ma[i] if not np.isnan(vol_ma[i]) else True
        
        # Price above/below 4h HMA confirmation
        price_above_hma = close[i] > hma_4h_aligned[i] * 1.001
        price_below_hma = close[i] < hma_4h_aligned[i] * 0.999
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Supertrend flip + 4h HMA bullish + RSI ok
        if st_long and fourh_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Supertrend bullish + 4h HMA bullish + RSI pullback (better entry)
        elif st_direction[i] == 1 and fourh_bullish and rsi_pullback_long and volume_ok:
            new_signal = SIZE_ENTRY
        # Strong trend (price well above 4h HMA) + Supertrend long
        elif price_above_hma and st_direction[i] == 1 and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        # Momentum continuation (RSI rising in bullish ST)
        elif st_direction[i] == 1 and fourh_bullish and rsi[i] > prev_rsi[i] and rsi[i] > 50:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Supertrend flip + 4h HMA bearish + RSI ok
        if st_short and fourh_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Supertrend bearish + 4h HMA bearish + RSI pullback (better entry)
        elif st_direction[i] == -1 and fourh_bearish and rsi_pullback_short and volume_ok:
            new_signal = -SIZE_ENTRY
        # Strong trend (price well below 4h HMA) + Supertrend short
        elif price_below_hma and st_direction[i] == -1 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Momentum continuation (RSI falling in bearish ST)
        elif st_direction[i] == -1 and fourh_bearish and rsi[i] < prev_rsi[i] and rsi[i] < 50:
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