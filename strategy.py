#!/usr/bin/env python3
"""
Experiment #413: 12h Supertrend + Daily HMA Bias + RSI Pullback + ATR Stop
Hypothesis: 12h timeframe captures multi-day swings while avoiding 1d's slow signal generation.
Combining 1d HMA for trend bias (HTF) with 12h Supertrend for direction and RSI pullbacks for
entry timing should generate MORE trades than 1d strategies while maintaining quality.
Key features: Multiple entry paths (Supertrend flip, RSI pullback, trend continuation),
1d HMA alignment via mtf_data helper (called ONCE), discrete sizing 0.25/0.30, 2.5*ATR stop.
Target: Beat Sharpe=0.499 with >=10 trades/symbol on train, >=3 on test.
Timeframe: 12h (REQUIRED), HTF: 1d for trend bias.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_supertrend_daily_hma_rsi_pullback_atr_v3"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator.
    Returns: supertrend_line, direction (1=bullish, -1=bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)
    direction[:] = np.nan
    
    # Calculate basic bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if np.isnan(atr[i]):
            continue
            
        # Bullish case: close above previous supertrend
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        # Bearish case: close below previous supertrend
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        # Continue previous trend
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
    
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    sma50 = calculate_sma(close, 50)
    sma200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma50[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (HTF direction)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # Supertrend flip signals (strongest entry signal)
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # RSI pullback conditions (enter on dip in uptrend, rally in downtrend)
        rsi_pullback_long = rsi[i] > 35 and rsi[i] < 55 and daily_bullish
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 65 and daily_bearish
        
        # RSI momentum (confirmation)
        rsi_momentum_long = rsi[i] > 45 and rsi[i] < 75
        rsi_momentum_short = rsi[i] > 25 and rsi[i] < 55
        
        # Price vs SMA filters
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        above_sma200 = close[i] > sma200[i]
        below_sma200 = close[i] < sma200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Supertrend flip + Daily bullish (strongest signal)
        if st_flip_long and daily_bullish:
            new_signal = SIZE_ENTRY
        # Path 2: Supertrend bullish + RSI pullback + Daily bullish
        elif st_bullish and rsi_pullback_long and daily_bullish:
            new_signal = SIZE_ENTRY
        # Path 3: Supertrend bullish + Daily bullish + RSI momentum + above SMA50
        elif st_bullish and daily_bullish and rsi_momentum_long and above_sma50:
            new_signal = SIZE_ENTRY
        # Path 4: Supertrend flip + above SMA50 (Daily neutral ok)
        elif st_flip_long and above_sma50 and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Path 5: Trend continuation - Daily bullish + Supertrend bullish + above SMA200
        elif daily_bullish and st_bullish and above_sma200 and rsi[i] > 50:
            new_signal = SIZE_ENTRY
        # Path 6: Simple momentum - Supertrend bullish + RSI > 50 + Daily bullish
        elif st_bullish and rsi[i] > 50 and daily_bullish and above_sma50:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Supertrend flip + Daily bearish (strongest signal)
        if st_flip_short and daily_bearish:
            new_signal = -SIZE_ENTRY
        # Path 2: Supertrend bearish + RSI pullback + Daily bearish
        elif st_bearish and rsi_pullback_short and daily_bearish:
            new_signal = -SIZE_ENTRY
        # Path 3: Supertrend bearish + Daily bearish + RSI momentum + below SMA50
        elif st_bearish and daily_bearish and rsi_momentum_short and below_sma50:
            new_signal = -SIZE_ENTRY
        # Path 4: Supertrend flip + below SMA50 (Daily neutral ok)
        elif st_flip_short and below_sma50 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 5: Trend continuation - Daily bearish + Supertrend bearish + below SMA200
        elif daily_bearish and st_bearish and below_sma200 and rsi[i] < 50:
            new_signal = -SIZE_ENTRY
        # Path 6: Simple momentum - Supertrend bearish + RSI < 50 + Daily bearish
        elif st_bearish and rsi[i] < 50 and daily_bearish and below_sma50:
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