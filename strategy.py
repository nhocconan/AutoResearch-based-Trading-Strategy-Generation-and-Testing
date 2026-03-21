#!/usr/bin/env python3
"""
Experiment #238: 4h Supertrend with Daily HMA Trend + Volume Momentum Filter
Hypothesis: 4h Supertrend (period=10, mult=2.5) captures multi-day trends better than 
12h for entry timing. Daily HMA(21) provides trend bias without being too restrictive. 
Volume confirmation (taker_buy_ratio > 0.55) filters false breakouts. ROC(10) momentum 
filter ensures we enter with trend strength. This combination should generate 30-50 
trades/year with 55%+ win rate. Position sizing: 0.25 entry, reduce to 0.125 at 2R.
Stoploss: 2.5*ATR trailing stop. Target: Beat Sharpe=0.499 from 12h Supertrend baseline.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_daily_hma_volume_roc_atr_v1"
timeframe = "4h"
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

def calculate_supertrend(high, low, close, period=10, multiplier=2.5):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower_band[0]
    
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            if direction[i-1] == 1:
                supertrend[i] = max(supertrend[i], lower_band[i])
            else:
                supertrend[i] = min(supertrend[i], upper_band[i])
    
    return supertrend, direction

def calculate_roc(close, period=10):
    """Calculate Rate of Change momentum indicator."""
    roc = np.zeros(len(close))
    for i in range(period, len(close)):
        roc[i] = (close[i] - close[i-period]) / close[i-period] * 100
    return roc

def calculate_taker_ratio(prices):
    """Calculate taker buy volume ratio."""
    if 'taker_buy_volume' in prices.columns:
        ratio = prices['taker_buy_volume'].values / (prices['volume'].values + 1e-10)
    else:
        ratio = np.ones(len(prices)) * 0.5
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, period=10, multiplier=2.5)
    roc = calculate_roc(close, 10)
    taker_ratio = calculate_taker_ratio(prices)
    
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
        # HTF trend filter (Daily HMA)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # Momentum filter (ROC)
        momentum_long = roc[i] > 0
        momentum_short = roc[i] < 0
        
        # Volume confirmation
        volume_long = taker_ratio[i] > 0.52
        volume_short = taker_ratio[i] < 0.48
        
        # Supertrend change detection
        st_change_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_change_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Supertrend flip to bullish with confirmations
        if st_change_long:
            if daily_bullish and momentum_long and volume_long:
                new_signal = SIZE_ENTRY
            elif daily_bullish and momentum_long:
                new_signal = SIZE_ENTRY * 0.8
        
        # Supertrend already bullish + pullback entry
        elif st_bullish and daily_bullish:
            # Enter on ROC dip but still positive
            if 0 < roc[i] < 5 and volume_long:
                if signals[i-1] == 0:
                    new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Supertrend flip to bearish with confirmations
        if st_change_short:
            if daily_bearish and momentum_short and volume_short:
                new_signal = -SIZE_ENTRY
            elif daily_bearish and momentum_short:
                new_signal = -SIZE_ENTRY * 0.8
        
        # Supertrend already bearish + pullback entry
        elif st_bearish and daily_bearish:
            # Enter on ROC bounce but still negative
            if -5 < roc[i] < 0 and volume_short:
                if signals[i-1] == 0:
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