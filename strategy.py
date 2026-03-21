#!/usr/bin/env python3
"""
Experiment #262: 4h Choppiness Regime + RSI Mean Reversion + Daily HMA Bias
Hypothesis: 2025 test period is bear/range market. Pure trend following fails (see #256).
Use Choppiness Index to detect regime: CHOP>61.8 = range (fade extremes), CHOP<38.2 = trend.
In range: RSI mean reversion at 30/70 levels. In trend: pullback entries with Daily HMA bias.
This adapts to changing market conditions better than fixed logic. Loose RSI thresholds
(25-75) ensure sufficient trades. Position sizing: 0.30 max, discrete levels to minimize churn.
Stoploss: 2.5*ATR trailing. Target: Beat Sharpe=0.499 from current best.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_rsi_daily_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (100*(sum(TR)/HH-LL)/ln(HH-LL))."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    sum_tr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    hhll = hh - ll
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * (sum_tr / hhll) / np.log(hhll)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
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

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

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
    chop = calculate_choppiness(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    
    # Previous values for crossover detection
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    prev_chop = np.roll(chop, 1)
    prev_chop[0] = chop[0]
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    position_reduced = False
    
    for i in range(60, n):
        # Daily trend bias
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Regime detection
        is_ranging = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # RSI levels (loose thresholds for more trades)
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_rising = rsi[i] > prev_rsi[i]
        rsi_falling = rsi[i] < prev_rsi[i]
        rsi_cross_up = prev_rsi[i] <= 45 and rsi[i] > 45
        rsi_cross_down = prev_rsi[i] >= 55 and rsi[i] < 55
        
        new_signal = 0.0
        
        # === RANGE REGIME: Mean Reversion ===
        if is_ranging:
            # Long: RSI oversold + starting to rise
            if rsi_oversold and rsi_rising:
                new_signal = SIZE
            # Short: RSI overbought + starting to fall
            elif rsi_overbought and rsi_falling:
                new_signal = -SIZE
        
        # === TREND REGIME: Trend Following ===
        elif is_trending:
            # Long: Daily bullish + RSI pullback
            if daily_bullish and 40 < rsi[i] < 60 and rsi_rising:
                new_signal = SIZE
            # Short: Daily bearish + RSI pullback
            elif daily_bearish and 40 < rsi[i] < 60 and rsi_falling:
                new_signal = -SIZE
            # Momentum long
            elif daily_bullish and rsi_cross_up:
                new_signal = SIZE
            # Momentum short
            elif daily_bearish and rsi_cross_down:
                new_signal = -SIZE
        
        # === NEUTRAL REGIME: Use Daily HMA Only ===
        else:
            # Simple bias-based entries
            if daily_bullish and rsi_oversold and rsi_rising:
                new_signal = SIZE
            elif daily_bearish and rsi_overbought and rsi_falling:
                new_signal = -SIZE
        
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
                    new_signal = SIZE / 2
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if lowest_close == 0.0 or close[i] < lowest_close:
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
                    new_signal = -SIZE / 2
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