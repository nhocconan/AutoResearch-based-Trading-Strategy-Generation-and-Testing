#!/usr/bin/env python3
"""
Experiment #316: 4h Supertrend + Daily HMA Bias + RSI Pullback + ATR Stops
Hypothesis: 4h Supertrend captures medium-term trends while Daily HMA provides
macro bias to filter counter-trend trades. RSI pullback entries (not breakouts)
reduce false signals and improve win rate. ATR trailing stops protect capital.
4h timeframe balances signal quality vs trade frequency - fewer trades than 1h
but more responsive than 12h/1d. Target: Beat Sharpe=0.499 with cleaner entries.
Timeframe: 4h (required for this experiment), HTF: 1d for trend bias.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_daily_hma_rsi_pullback_atr_v3"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    n = len(close)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    final_upper[0] = upper_band[0]
    final_lower[0] = lower_band[0]
    supertrend[0] = lower_band[0]
    
    for i in range(1, n):
        # Calculate final upper/lower bands
        if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i-1]
        
        if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i-1]
        
        # Determine trend and supertrend value
        if trend[i-1] == 1:
            if close[i] < final_lower[i]:
                trend[i] = -1
                supertrend[i] = final_upper[i]
            else:
                trend[i] = 1
                supertrend[i] = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                trend[i] = 1
                supertrend[i] = final_lower[i]
            else:
                trend[i] = -1
                supertrend[i] = final_upper[i]
    
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
    rsi = calculate_rsi(close, 14)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    sma_200 = calculate_sma(close, 200)
    
    # Track previous values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    prev_st_trend = np.roll(st_trend, 1)
    prev_st_trend[0] = st_trend[0]
    
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
    
    for i in range(250, n):  # Start after SMA200 is ready
        # Skip if indicators not ready
        if np.isnan(supertrend[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Daily macro trend bias
        daily_bullish = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        daily_bearish = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # Long-term trend filter
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # Supertrend signals
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # Supertrend flip (entry signal)
        st_flip_long = st_bullish and prev_st_trend[i] == -1
        st_flip_short = st_bearish and prev_st_trend[i] == 1
        
        # RSI pullback zones (enter on dips in trend, not breakouts)
        rsi_pullback_long = 35 < rsi[i] < 55  # Dip in uptrend
        rsi_pullback_short = 45 < rsi[i] < 65  # Rally in downtrend
        rsi_momentum_long = rsi[i] > 50
        rsi_momentum_short = rsi[i] < 50
        
        # Price above/below supertrend
        above_st = close[i] > supertrend[i]
        below_st = close[i] < supertrend[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Daily bullish + Supertrend flip + RSI pullback
        if daily_bullish and st_flip_long and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        # Secondary: Daily bullish + Supertrend bullish + RSI momentum + Above SMA200
        elif daily_bullish and st_bullish and rsi_momentum_long and above_sma200:
            new_signal = SIZE_ENTRY
        # Tertiary: Supertrend flip + Above SMA200 + RSI 40-60 (momentum)
        elif st_flip_long and above_sma200 and 40 < rsi[i] < 60:
            new_signal = SIZE_ENTRY
        # Quaternary: Above supertrend + Daily bullish + RSI > 45 (simple trend)
        elif above_st and daily_bullish and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: Daily bearish + Supertrend flip + RSI pullback
        if daily_bearish and st_flip_short and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Daily bearish + Supertrend bearish + RSI momentum + Below SMA200
        elif daily_bearish and st_bearish and rsi_momentum_short and below_sma200:
            new_signal = -SIZE_ENTRY
        # Tertiary: Supertrend flip + Below SMA200 + RSI 40-60 (momentum)
        elif st_flip_short and below_sma200 and 40 < rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Quaternary: Below supertrend + Daily bearish + RSI < 55 (simple trend)
        elif below_st and daily_bearish and rsi[i] < 55:
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