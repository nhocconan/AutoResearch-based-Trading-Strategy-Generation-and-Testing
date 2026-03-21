#!/usr/bin/env python3
"""
Experiment #273: 1h Supertrend + 4h HMA Trend + RSI Pullback with Volume Confirmation
Hypothesis: Simpler is better. The winning strategy (mtf_12h_supertrend_daily_hma_rsi_pullback_v2)
uses Supertrend + HMA trend + RSI pullback. This adapts it to 1h primary with 4h HMA trend filter.
Key differences: (1) 1h timeframe for more responsive entries, (2) 4h HMA for trend bias (not daily),
(3) RSI pullback zone 35-65 (not extremes), (4) Volume ratio confirmation >0.55 for longs, <0.45 for shorts.
Position sizing: 0.28 entry, 0.14 half at 2R profit. Stoploss: 2.5*ATR trailing. Target: Beat Sharpe=0.499.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_supertrend_4h_hma_rsi_pullback_volume_atr_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator (trend direction and levels)."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    trend = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper_band[0]
    trend[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            trend[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper_band[i]
            trend[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            trend[i] = trend[i-1]
    
    return supertrend, trend

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
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

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (0-1, >0.5 = bullish)."""
    ratio = np.where(volume > 0, taker_buy_volume / volume, 0.5)
    return ratio

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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
    # Track previous values for signal changes
    prev_st_trend = np.roll(st_trend, 1)
    prev_st_trend[0] = st_trend[0]
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
        # HTF trend filter (4h HMA)
        hma_bullish = close[i] > hma_4h_aligned[i]
        hma_bearish = close[i] < hma_4h_aligned[i]
        
        # Supertrend signals
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        st_cross_up = prev_st_trend[i] == -1 and st_trend[i] == 1
        st_cross_down = prev_st_trend[i] == 1 and st_trend[i] == -1
        
        # RSI pullback zone (not extreme, but confirming momentum)
        rsi_pullback_long = 35 < rsi[i] < 60
        rsi_pullback_short = 40 < rsi[i] < 65
        rsi_momentum_long = rsi[i] > prev_rsi[i] and rsi[i] > 45
        rsi_momentum_short = rsi[i] < prev_rsi[i] and rsi[i] < 55
        
        # Volume confirmation
        vol_bullish = vol_ratio[i] > 0.55
        vol_bearish = vol_ratio[i] < 0.45
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Supertrend cross up with trend confirmation
        if st_cross_up:
            if hma_bullish and rsi_pullback_long:
                new_signal = SIZE_ENTRY
            elif hma_bullish and vol_bullish:
                new_signal = SIZE_ENTRY
        
        # Supertrend already bullish + pullback entry
        elif st_bullish and hma_bullish:
            if rsi_momentum_long and vol_bullish:
                new_signal = SIZE_ENTRY
            elif prev_st_trend[i] == 1 and rsi_pullback_long:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Supertrend cross down with trend confirmation
        if st_cross_down:
            if hma_bearish and rsi_pullback_short:
                new_signal = -SIZE_ENTRY
            elif hma_bearish and vol_bearish:
                new_signal = -SIZE_ENTRY
        
        # Supertrend already bearish + pullback entry
        elif st_bearish and hma_bearish:
            if rsi_momentum_short and vol_bearish:
                new_signal = -SIZE_ENTRY
            elif prev_st_trend[i] == -1 and rsi_pullback_short:
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