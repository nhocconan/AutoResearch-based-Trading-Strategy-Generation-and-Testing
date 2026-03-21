#!/usr/bin/env python3
"""
Experiment #459: 1h Fisher Transform Reversals + 4h HMA Trend + ATR Stop
Hypothesis: Fisher Transform catches reversal turning points better than RSI in bear/range markets.
Combined with 4h HMA trend filter for direction bias, this should work on 1h timeframe.
Fisher crosses -1.5 (long) or +1.5 (short) with trend confirmation = high probability entries.
Multiple entry paths ensure >=10 trades requirement is met. Loose Fisher thresholds (-1.2/+1.2).
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_hma_trend_reversal_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches turning points better than RSI in ranging markets.
    Reference: Ehlers, J.F. "Rocket Science for Traders"
    """
    close_s = pd.Series(close)
    # Calculate highest high and lowest low over period
    hh = close_s.rolling(window=period, min_periods=period).max().values
    ll = close_s.rolling(window=period, min_periods=period).min().values
    
    # Normalize price to range -1 to +1
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 0.001, range_hl)  # avoid division by zero
    normalized = 0.66 * ((close - ll) / range_hl - 0.5) + 0.67 * np.roll(normalized, 1) if len(close) > 1 else np.zeros(len(close))
    
    # Proper Fisher calculation
    fisher = np.zeros(len(close))
    fisher[:] = np.nan
    
    for i in range(period, len(close)):
        if range_hl[i] > 0:
            val = 0.66 * ((close[i] - ll[i]) / range_hl[i] - 0.5) + 0.67 * (normalized[i-1] if i > 0 else 0)
            val = np.clip(val, -0.99, 0.99)  # prevent log errors
            fisher[i] = 0.5 * np.log((1 + val) / (1 - val))
    
    return fisher

def calculate_fisher_signal(close, period=9):
    """
    Simplified Fisher Transform with signal line for cleaner crossovers.
    """
    close_s = pd.Series(close)
    hh = close_s.rolling(window=period, min_periods=period).max().values
    ll = close_s.rolling(window=period, min_periods=period).min().values
    
    range_hl = hh - ll
    range_hl = np.where(range_hl < 0.0001, 0.0001, range_hl)
    
    fisher = np.zeros(len(close))
    fisher[:] = np.nan
    
    prev_val = 0.0
    for i in range(period, len(close)):
        normalized = 0.66 * ((close[i] - ll[i]) / range_hl[i] - 0.5) + 0.67 * prev_val
        normalized = np.clip(normalized, -0.99, 0.99)
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        prev_val = normalized
    
    return fisher

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

def calculate_slope(values, lookback=5):
    """Calculate slope of values over lookback period."""
    n = len(values)
    slope = np.zeros(n)
    slope[:] = np.nan
    for i in range(lookback, n):
        if not np.isnan(values[i]) and not np.isnan(values[i - lookback]):
            slope[i] = (values[i] - values[i - lookback]) / lookback
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    hma_1h = calculate_hma(close, 21)
    hma_1h_fast = calculate_hma(close, 9)
    rsi = calculate_rsi(close, 14)
    fisher = calculate_fisher_signal(close, 9)
    hma_slope = calculate_slope(hma_1h, lookback=5)
    
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
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1h[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(hma_slope[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h HMA trend
        hma_1h_bullish = close[i] > hma_1h[i]
        hma_1h_bearish = close[i] < hma_1h[i]
        hma_rising = hma_slope[i] > 0
        hma_falling = hma_slope[i] < 0
        
        # Fast HMA crossover
        fast_above_slow = hma_1h_fast[i] > hma_1h[i]
        fast_below_slow = hma_1h_fast[i] < hma_1h[i]
        
        # Fisher Transform signals (loose thresholds for more trades)
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        fisher_cross_up = fisher[i] > -1.2 and fisher[i-1] <= -1.2 if i > 0 else False
        fisher_cross_down = fisher[i] < 1.2 and fisher[i-1] >= 1.2 if i > 0 else False
        
        # RSI zones
        rsi_neutral = rsi[i] > 35 and rsi[i] < 65
        rsi_bullish = rsi[i] > 45
        rsi_bearish = rsi[i] < 55
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: 4h bullish + Fisher cross up + HMA rising
        if trend_bullish and fisher_cross_up and hma_rising:
            new_signal = SIZE_ENTRY
        # Path 2: 4h bullish + 1h bullish + Fisher oversold (reversal)
        elif trend_bullish and hma_1h_bullish and fisher_oversold:
            new_signal = SIZE_ENTRY
        # Path 3: Fast HMA above slow + Fisher cross up + RSI > 40
        elif fast_above_slow and fisher_cross_up and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Path 4: 4h bullish + Fast HMA crossover + RSI neutral
        elif trend_bullish and fast_above_slow and rsi_neutral:
            new_signal = SIZE_ENTRY
        # Path 5: Price above both HMA + Fisher < -0.5 (pullback entry)
        elif close[i] > hma_1h[i] and close[i] > hma_4h_aligned[i] and fisher[i] < -0.5:
            new_signal = SIZE_ENTRY
        # Path 6: HMA rising + Fisher cross up (momentum continuation)
        elif hma_rising and fisher_cross_up and rsi_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: 4h bearish + Fisher cross down + HMA falling
        if trend_bearish and fisher_cross_down and hma_falling:
            new_signal = -SIZE_ENTRY
        # Path 2: 4h bearish + 1h bearish + Fisher overbought (reversal)
        elif trend_bearish and hma_1h_bearish and fisher_overbought:
            new_signal = -SIZE_ENTRY
        # Path 3: Fast HMA below slow + Fisher cross down + RSI < 60
        elif fast_below_slow and fisher_cross_down and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 4: 4h bearish + Fast HMA crossover + RSI neutral
        elif trend_bearish and fast_below_slow and rsi_neutral:
            new_signal = -SIZE_ENTRY
        # Path 5: Price below both HMA + Fisher > 0.5 (rally short)
        elif close[i] < hma_1h[i] and close[i] < hma_4h_aligned[i] and fisher[i] > 0.5:
            new_signal = -SIZE_ENTRY
        # Path 6: HMA falling + Fisher cross down (momentum continuation)
        elif hma_falling and fisher_cross_down and rsi_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 1h timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for 1h timeframe)
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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