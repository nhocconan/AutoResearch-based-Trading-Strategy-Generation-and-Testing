#!/usr/bin/env python3
"""
Experiment #382: 4h EMA Crossover + Daily HMA Trend + RSI Pullback + ATR Stop
Hypothesis: Simpler is better for 4h timeframe. Previous complex strategies (KAMA, CHOP, Supertrend)
failed on 4h due to over-filtering. This strategy uses proven components: EMA(9/21) crossover for
entry timing, Daily HMA(21) for trend bias (proven in best 12h strategy), RSI pullback zone (40-60)
for entry quality, and 2.0x ATR trailing stop. Key insight: remove CHOP regime filter (caused whipsaws
in 340+ failed strategies), use fewer but higher-quality filters. 4h captures medium-term trends
without the noise of lower timeframes. Position sizing: 0.25 entry, 0.125 half at 2R profit.
Target: Beat Sharpe=0.499 with 30-60 trades total, DD < -30%.
Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_ema_daily_hma_rsi_pullback_atr_v1"
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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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
    ema_fast = calculate_ema(close, 9)
    ema_slow = calculate_ema(close, 21)
    
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
    
    for i in range(50, n):  # Start after 50 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias
        daily_bullish = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        daily_bearish = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # EMA crossover signals
        ema_cross_long = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_short = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # EMA position (already crossed)
        ema_bullish = ema_fast[i] > ema_slow[i]
        ema_bearish = ema_fast[i] < ema_slow[i]
        
        # RSI pullback zone (40-60 for quality entries)
        rsi_pullback_long = rsi[i] >= 40 and rsi[i] <= 60
        rsi_pullback_short = rsi[i] >= 40 and rsi[i] <= 60
        
        # RSI momentum (not extreme)
        rsi_ok_long = rsi[i] > 35 and rsi[i] < 70
        rsi_ok_short = rsi[i] > 30 and rsi[i] < 65
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: EMA cross long + Daily bullish + RSI pullback
        if ema_cross_long and daily_bullish and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        # Secondary: EMA cross long + Daily bullish + RSI ok (looser)
        elif ema_cross_long and daily_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Tertiary: EMA bullish + Daily bullish + RSI pullback (continuation)
        elif ema_bullish and daily_bullish and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        # Quaternary: EMA cross long + RSI ok (daily neutral, strong momentum)
        elif ema_cross_long and rsi[i] > 45 and rsi[i] < 65:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: EMA cross short + Daily bearish + RSI pullback
        if ema_cross_short and daily_bearish and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        # Secondary: EMA cross short + Daily bearish + RSI ok (looser)
        elif ema_cross_short and daily_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: EMA bearish + Daily bearish + RSI pullback (continuation)
        elif ema_bearish and daily_bearish and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        # Quaternary: EMA cross short + RSI ok (daily neutral, strong momentum)
        elif ema_cross_short and rsi[i] > 35 and rsi[i] < 55:
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