#!/usr/bin/env python3
"""
Experiment #173: 12h Supertrend with Daily/Weekly HMA Trend Filter
Hypothesis: Simpler is better. Complex regime detection (#161) failed with Sharpe=-0.057.
The current best (Sharpe=0.499) uses Supertrend + Daily HMA + RSI pullback.
This version: 12h Supertrend for primary signal, Daily HMA for trend bias,
Weekly HMA for macro direction confirmation, RSI for pullback timing.
Key changes from #161: Remove Choppiness/BB regime detection (too many filters = few trades),
use cleaner Supertrend signals, simpler position tracking, looser RSI thresholds (35/65).
Target: More trades (>50/train), positive Sharpe on ALL symbols, DD < -30%.
Position sizing: 0.25 entry, 0.15 half at 2R profit, stoploss at 2.5*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_supertrend_daily_weekly_hma_rsi_v1"
timeframe = "12h"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_line, supertrend_direction (1=bullish, -1=bearish)
    Reference: Seby Airan, TradingView
    """
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    n = len(close)
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish (price above ST)
    
    supertrend[0] = upper_band[0]
    
    for i in range(1, n):
        if direction[i-1] == 1:
            # Previously bullish
            if close[i] > lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            # Previously bearish
            if close[i] < upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF HMAs
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.15
    
    # Position tracking
    position_side = 0
    entry_price = 0.0
    stoploss_price = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters (loose - just bias, not strict requirement)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # 12h trend
        trend_bullish = hma_21[i] > hma_50[i]
        trend_bearish = hma_21[i] < hma_50[i]
        
        # Supertrend signals
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # RSI signals (looser thresholds for more trades)
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        rsi_rising = rsi[i] > rsi[i-2] if i > 2 else False
        rsi_falling = rsi[i] < rsi[i-2] if i > 2 else False
        
        # Supertrend direction change (entry signal)
        st_turned_bullish = st_bullish and (i > 0 and st_direction[i-1] == -1)
        st_turned_bearish = st_bearish and (i > 0 and st_direction[i-1] == 1)
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Supertrend turn bullish + daily trend supportive
        if st_turned_bullish:
            if daily_bullish or weekly_bullish:
                new_signal = SIZE_ENTRY
            elif rsi_oversold and rsi_rising:
                # Counter-trend long on oversold RSI
                new_signal = SIZE_ENTRY
        
        # Pullback entry: trend bullish + RSI pullback + Supertrend still bullish
        elif st_bullish and trend_bullish:
            if rsi[i] < 50 and rsi_rising:
                if daily_bullish:
                    new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: Supertrend turn bearish + daily trend supportive
        if st_turned_bearish:
            if daily_bearish or weekly_bearish:
                new_signal = -SIZE_ENTRY
            elif rsi_overbought and rsi_falling:
                # Counter-trend short on overbought RSI
                new_signal = -SIZE_ENTRY
        
        # Pullback entry: trend bearish + RSI pullback + Supertrend still bearish
        elif st_bearish and trend_bearish:
            if rsi[i] > 50 and rsi_falling:
                if daily_bearish:
                    new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Trailing stop: 2.5*ATR from highest
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > stoploss_price:
                stoploss_price = current_stop
            
            # Check stoploss hit
            if close[i] < stoploss_price:
                new_signal = 0.0
            # Take profit at 2R
            elif not position_reduced:
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            # Trailing stop: 2.5*ATR from lowest
            current_stop = lowest_close + 2.5 * atr[i]
            if stoploss_price == 0.0 or current_stop < stoploss_price:
                stoploss_price = current_stop
            
            # Check stoploss hit
            if close[i] > stoploss_price:
                new_signal = 0.0
            # Take profit at 2R
            elif not position_reduced:
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            stoploss_price = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            stoploss_price = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
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
            stoploss_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals