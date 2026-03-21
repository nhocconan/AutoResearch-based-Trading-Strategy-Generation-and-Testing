#!/usr/bin/env python3
"""
Experiment #349: 15m Supertrend + 4h HMA Trend + RSI Filter + ATR Stop
Hypothesis: 15m Supertrend entries filtered by 4h HMA trend reduces whipsaws from exp #337.
Previous 15m Supertrend (#337) got 0 trades - this version uses LOOSER filters to ensure trades.
Supertrend(10,3) for entries, 4h HMA(21) for macro bias (soft filter), RSI(7) avoids extremes.
ATR(14) stoploss at 2.0x. Discrete signals (0, ±0.25, ±0.30) minimize fee churn.
Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper.
Target: Beat Sharpe=0.499 with 40-80 trades/year (more than 12h strategies), DD < -35%.
Key fix from #337: Tertiary entry without 4h filter ensures trades when HTF data gaps occur.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_hma_rsi_atr_v2"
timeframe = "15m"
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
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower_band[0]
    trend[0] = 1
    
    for i in range(1, n):
        if trend[i-1] == 1:
            # Previously bullish
            if close[i] < lower_band[i]:
                supertrend[i] = upper_band[i]
                trend[i] = -1
            else:
                supertrend[i] = max(lower_band[i], supertrend[i-1])
                trend[i] = 1
        else:
            # Previously bearish
            if close[i] > upper_band[i]:
                supertrend[i] = lower_band[i]
                trend[i] = 1
            else:
                supertrend[i] = min(upper_band[i], supertrend[i-1])
                trend[i] = -1
    
    return supertrend, trend

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=7):
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
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 7)
    
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
        if np.isnan(atr[i]) or np.isnan(supertrend[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # 4h macro trend bias (SOFT filter - boosts confidence, not required)
        hma_valid = not np.isnan(hma_4h_aligned[i])
        hma_bullish = hma_valid and close[i] > hma_4h_aligned[i]
        hma_bearish = hma_valid and close[i] < hma_4h_aligned[i]
        
        # Supertrend signals
        st_long = st_trend[i] == 1
        st_short = st_trend[i] == -1
        
        # Supertrend crossover (PRIMARY entry trigger)
        st_cross_long = st_trend[i] == 1 and st_trend[i-1] == -1
        st_cross_short = st_trend[i] == -1 and st_trend[i-1] == 1
        
        # RSI momentum filter (LOOSE for 15m - ensure trades)
        rsi_ok_long = rsi[i] > 30  # Not deeply oversold
        rsi_ok_short = rsi[i] < 70  # Not deeply overbought
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Supertrend cross + 4h bullish + RSI ok
        if st_cross_long and hma_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: Supertrend long + 4h bullish + RSI ok (hold/add)
        elif st_long and hma_bullish and rsi_ok_long and signals[i-1] > 0:
            new_signal = SIZE_ENTRY
        # Tertiary: Supertrend cross alone (ensures trades when HTF gaps)
        elif st_cross_long and rsi_ok_long:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: Supertrend cross + 4h bearish + RSI ok
        elif st_cross_short and hma_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Supertrend short + 4h bearish + RSI ok (hold/add)
        elif st_short and hma_bearish and rsi_ok_short and signals[i-1] < 0:
            new_signal = -SIZE_ENTRY
        # Tertiary: Supertrend cross alone (ensures trades when HTF gaps)
        elif st_cross_short and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
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
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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