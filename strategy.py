#!/usr/bin/env python3
"""
Experiment #354: 1d Donchian Breakout + Weekly HMA + RSI Filter + ATR Stop
Hypothesis: Daily Donchian channels (20-day) capture trend breakouts effectively on crypto.
Weekly HMA provides macro trend bias to filter counter-trend breakouts. RSI(14) with
loose thresholds (25-75) ensures sufficient trade frequency. ATR(14) stoploss at 2.5x
protects during reversals. This combines breakout momentum with trend confirmation.
Timeframe: 1d (REQUIRED), HTF: 1w for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with 30-60 trades total across train+test.
Key insight: Donchian breakouts work well on daily crypto data; weekly filter reduces whipsaws.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_hma_rsi_breakout_atr_v1"
timeframe = "1d"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Calculate Donchian midpoint
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
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
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        # Weekly macro trend bias (SOFT filter - boosts confidence)
        weekly_bullish = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        weekly_bearish = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # Price position relative to Donchian channel
        above_mid = close[i] > donchian_mid[i]
        below_mid = close[i] < donchian_mid[i]
        
        # RSI momentum filter (LOOSE for daily - ensure trades)
        rsi_ok_long = rsi[i] > 25  # Not deeply oversold
        rsi_ok_short = rsi[i] < 75  # Not deeply overbought
        
        # RSI confirmation
        rsi_strong_long = rsi[i] > 35
        rsi_strong_short = rsi[i] < 65
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Donchian breakout + Weekly bullish + RSI ok
        if breakout_long and weekly_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: Price above mid + Weekly bullish + RSI strong
        elif above_mid and weekly_bullish and rsi_strong_long:
            new_signal = SIZE_ENTRY
        # Tertiary: Donchian breakout without weekly filter (momentum only - ensures trades)
        elif breakout_long and rsi[i] > 30:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: Donchian breakout + Weekly bearish + RSI ok
        if breakout_short and weekly_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Price below mid + Weekly bearish + RSI strong
        elif below_mid and weekly_bearish and rsi_strong_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: Donchian breakout without weekly filter (momentum only - ensures trades)
        elif breakout_short and rsi[i] < 70:
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