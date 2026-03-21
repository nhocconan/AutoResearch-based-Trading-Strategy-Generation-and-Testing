#!/usr/bin/env python3
"""
Experiment #337: 15m Supertrend + Dual HTF HMA Filter (4h+1h) + RSI Momentum + ATR Stop
Hypothesis: 15m timeframe failed in exp#325 (Sharpe=-7.403) with mean reversion.
This attempt uses TREND FOLLOWING with DUAL HTF filter (4h HMA + 1h HMA must agree).
Previous 15m failures used single HTF or no HTF filter. Dual agreement should reduce
false signals by 50%+ while maintaining sufficient trade frequency (30-50/year).
Supertrend(10,3) for entry timing, RSI(14) 40-60 filter for trend continuation,
2.5*ATR trailing stop. Position size 0.25 (25% capital) to limit drawdown.
Key insight: 15m needs STRONGER HTF filter than 1h/4h strategies to overcome noise.
Target: Beat Sharpe=0.499 with clean trend-following, DD < -30%, trades >= 10/train.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_dual_hma_4h_1h_rsi_atr_v1"
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
    """Calculate Supertrend indicator (trend following)."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    
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
        # Update upper band
        if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i-1]
        
        # Update lower band
        if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i-1]
        
        # Determine trend
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    
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
    
    for i in range(250, n):  # Start after 250 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        # Skip if HTF data not available
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Dual HTF trend filter (BOTH must agree)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        hma_1h_bullish = close[i] > hma_1h_aligned[i]
        hma_1h_bearish = close[i] < hma_1h_aligned[i]
        
        # Both HTFs must agree for strong signal
        dual_bullish = hma_4h_bullish and hma_1h_bullish
        dual_bearish = hma_4h_bearish and hma_1h_bearish
        
        # Supertrend signal
        st_long = st_trend[i] == 1
        st_short = st_trend[i] == -1
        
        # RSI momentum filter (trend continuation, not extreme)
        rsi_ok_long = 40 <= rsi[i] <= 65  # Not overbought for long
        rsi_ok_short = 35 <= rsi[i] <= 60  # Not oversold for short
        
        # RSI strength filter
        rsi_strong_long = rsi[i] > 50
        rsi_strong_short = rsi[i] < 50
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Supertrend long + Dual HTF bullish + RSI ok
        if st_long and dual_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: Supertrend long + 4h bullish only + RSI strong
        elif st_long and hma_4h_bullish and rsi_strong_long and rsi[i] < 60:
            new_signal = SIZE_ENTRY
        # Tertiary: Supertrend flip to long + Dual HTF bullish
        if i > 0 and st_trend[i-1] == -1 and st_long and dual_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: Supertrend short + Dual HTF bearish + RSI ok
        if st_short and dual_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Supertrend short + 4h bearish only + RSI strong
        elif st_short and hma_4h_bearish and rsi_strong_short and rsi[i] > 40:
            new_signal = -SIZE_ENTRY
        # Tertiary: Supertrend flip to short + Dual HTF bearish
        if i > 0 and st_trend[i-1] == 1 and st_short and dual_bearish:
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