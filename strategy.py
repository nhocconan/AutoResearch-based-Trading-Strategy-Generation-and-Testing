#!/usr/bin/env python3
"""
Experiment #368: 30m Supertrend + 4h HMA Trend + RSI Momentum + ATR Stop
Hypothesis: 30m timeframe offers better risk/reward than 12h (faster entries) while avoiding
the noise of 15m. 4h HMA provides reliable trend bias without being too slow. Supertrend(10,3)
gives clear entry/exit signals. RSI(14) with loose thresholds (30-70) ensures trade frequency.
ATR(14) stoploss at 2.2x protects capital during reversals. Building on #359's success with
multi-timeframe approach but optimizing for 30m's sweet spot between speed and reliability.
Timeframe: 30m (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with 40-80 trades total across train+test.
Key insight: 30m captures intraday swings while 4h filter avoids counter-trend traps.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_supertrend_4h_hma_rsi_momentum_atr_v1"
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
    """Calculate Supertrend indicator with direction."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish (price above ST), -1 = bearish
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
        
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        else:
            # Upper band logic
            if close[i - 1] <= supertrend[i - 1]:
                upper_band[i] = min(upper_band[i], supertrend[i - 1])
            else:
                upper_band[i] = upper_band[i]
            
            # Lower band logic
            if close[i - 1] >= supertrend[i - 1]:
                lower_band[i] = max(lower_band[i], supertrend[i - 1])
            else:
                lower_band[i] = lower_band[i]
            
            # Supertrend value and direction
            if close[i] <= supertrend[i - 1]:
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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
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
        
        # 4h trend bias
        hma_4h_valid = not np.isnan(hma_4h_aligned[i])
        trend_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # Supertrend signals
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend direction change (entry trigger)
        st_change_long = i > 0 and st_direction[i] == 1 and st_direction[i-1] == -1
        st_change_short = i > 0 and st_direction[i] == -1 and st_direction[i-1] == 1
        
        # RSI momentum filter (LOOSE for trade frequency)
        rsi_ok_long = rsi[i] > 30 and rsi[i] < 75
        rsi_ok_short = rsi[i] > 25 and rsi[i] < 70
        
        # RSI confirmation for stronger signals
        rsi_strong_long = rsi[i] > 40 and rsi[i] < 70
        rsi_strong_short = rsi[i] > 30 and rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Supertrend flip long + 4h bullish + RSI ok
        if st_change_long and trend_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: Supertrend already long + 4h bullish + RSI strong (continuation)
        elif st_long and trend_bullish and rsi_strong_long and position_side == 0:
            new_signal = SIZE_ENTRY
        # Tertiary: Supertrend flip long alone (ensures minimum trade frequency)
        elif st_change_long and rsi[i] > 35:
            new_signal = SIZE_ENTRY
        # Quaternary: Price above 4h HMA + Supertrend long (trend following)
        elif trend_bullish and st_long and rsi[i] > 35 and rsi[i] < 70 and position_side == 0:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: Supertrend flip short + 4h bearish + RSI ok
        if st_change_short and trend_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Supertrend already short + 4h bearish + RSI strong (continuation)
        elif st_short and trend_bearish and rsi_strong_short and position_side == 0:
            new_signal = -SIZE_ENTRY
        # Tertiary: Supertrend flip short alone (ensures minimum trade frequency)
        elif st_change_short and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Quaternary: Price below 4h HMA + Supertrend short (trend following)
        elif trend_bearish and st_short and rsi[i] > 30 and rsi[i] < 65 and position_side == 0:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.2*ATR from highest)
            current_stop = highest_close - 2.2 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.2 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.2*ATR from lowest)
            current_stop = lowest_close + 2.2 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.2 * atr[i]
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
            trailing_stop = close[i] - 2.2 * atr[i] if position_side > 0 else close[i] + 2.2 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.2 * atr[i] if position_side > 0 else close[i] + 2.2 * atr[i]
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