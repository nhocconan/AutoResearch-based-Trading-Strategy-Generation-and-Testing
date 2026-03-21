#!/usr/bin/env python3
"""
Experiment #384: 1d Supertrend + Weekly HMA Trend + RSI Pullback + ATR Stop
Hypothesis: Daily timeframe captures major trends with fewer whipsaws than intraday.
Supertrend(10,3) provides clear trend direction with ATR-based stops. Weekly HMA(21)
gives major trend bias to avoid counter-trend trades. RSI(14) with moderate thresholds
(30-70) ensures entries on pullbacks within trend, not at extremes. This combination
should produce fewer but higher-quality trades than 12h strategies. Position size 0.25
discrete to minimize fee churn while maintaining exposure. ATR(14) stoploss at 2.5x
protects against major reversals. Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper.
Target: Beat Sharpe=0.499 (current best mtf_12h_supertrend_daily_hma_rsi_pullback_v2).
Key insight: Daily bars + weekly bias = cleaner trends, fewer false signals than 12h.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_supertrend_weekly_hma_rsi_pullback_atr_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend values, direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    # Calculate HL2 and basic upper/lower bands
    hl2 = (high + low) / 2.0
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
        
        # Initial bands
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        # Smooth bands (can't go below previous lower / above previous upper)
        if i > period:
            if upper_band[i] < upper_band[i-1] or close[i-1] > upper_band[i-1]:
                upper_band[i] = hl2[i] + multiplier * atr[i]
            else:
                upper_band[i] = upper_band[i-1]
            
            if lower_band[i] > lower_band[i-1] or close[i-1] < lower_band[i-1]:
                lower_band[i] = hl2[i] - multiplier * atr[i]
            else:
                lower_band[i] = lower_band[i-1]
        
        # Determine direction and supertrend value
        if i == period:
            direction[i] = 1
            supertrend[i] = lower_band[i]
        else:
            if direction[i-1] == 1:
                if close[i] < lower_band[i]:
                    direction[i] = -1
                    supertrend[i] = upper_band[i]
                else:
                    direction[i] = 1
                    supertrend[i] = lower_band[i]
            else:
                if close[i] > upper_band[i]:
                    direction[i] = 1
                    supertrend[i] = lower_band[i]
                else:
                    direction[i] = -1
                    supertrend[i] = upper_band[i]
    
    supertrend[:period] = np.nan
    direction[:period] = 0
    return supertrend, direction

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
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
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias
        weekly_bullish = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        weekly_bearish = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip signals
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # RSI pullback levels (moderate, not extreme)
        rsi_pullback_long = rsi[i] > 35 and rsi[i] < 65  # Pullback in uptrend
        rsi_pullback_short = rsi[i] > 35 and rsi[i] < 65  # Pullback in downtrend
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 40 and rsi[i] < 70
        rsi_momentum_short = rsi[i] > 30 and rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Supertrend flip long + Weekly bullish + RSI ok
        if st_flip_long and weekly_bullish and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Secondary: Supertrend long + Weekly bullish + RSI pullback (entry on dip)
        elif st_long and weekly_bullish and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        # Tertiary: Supertrend flip long alone (ensures trade frequency)
        elif st_flip_long and rsi[i] > 35:
            new_signal = SIZE_ENTRY
        # Quaternary: Supertrend long + RSI momentum (no weekly filter)
        elif st_long and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: Supertrend flip short + Weekly bearish + RSI ok
        if st_flip_short and weekly_bearish and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Supertrend short + Weekly bearish + RSI pullback (entry on rally)
        elif st_short and weekly_bearish and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: Supertrend flip short alone (ensures trade frequency)
        elif st_flip_short and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Quaternary: Supertrend short + RSI momentum (no weekly filter)
        elif st_short and rsi_momentum_short:
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