#!/usr/bin/env python3
"""
Experiment #060: 1d Donchian Breakout with Weekly HMA Trend Bias + Volume Filter
Hypothesis: Daily timeframe captures major trends while avoiding noise. Donchian 20-period
breakouts are classic trend-following signals that work well on daily data. Weekly HMA(21)
provides trend bias without being overly restrictive. Volume confirmation ensures breakouts
have participation. Simple 2.5*ATR trailing stop. Position sizing 0.30 entry, 0.15 at 2R.
This should generate 15-30 trades/year on daily data while capturing major moves.
Key insight from failures: fewer filters = more trades = better chance of positive Sharpe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_hma_volume_v1"
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
    half = period // 2
    if half < 1:
        half = 1
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel breakout levels."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Simple trend filter - 50 day HMA
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    
    for i in range(100, n):
        # Weekly trend bias (HTF) - only used as soft filter
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # Volume confirmation (must be above average)
        volume_confirmed = volume[i] > vol_ma[i]
        
        # 50-day trend filter
        trend_long = close[i] > hma_50[i]
        trend_short = close[i] < hma_50[i]
        
        new_signal = 0.0
        
        # LONG ENTRY: Donchian breakout + volume + trend alignment
        # Weekly bias is soft - we can enter against it if breakout is strong
        if breakout_long and volume_confirmed:
            if weekly_bullish or trend_long:
                new_signal = SIZE_ENTRY
            elif not weekly_bearish:  # neutral weekly, but breakout + volume
                new_signal = SIZE_ENTRY * 0.67  # reduced size
        
        # SHORT ENTRY: Donchian breakout + volume + trend alignment
        if breakout_short and volume_confirmed:
            if weekly_bearish or trend_short:
                new_signal = -SIZE_ENTRY
            elif not weekly_bullish:  # neutral weekly, but breakout + volume
                new_signal = -SIZE_ENTRY * 0.67  # reduced size
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Calculate trailing stop
            current_stop = close[i] - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = close[i] - entry_price
                    risk = 2.5 * atr[i]  # initial risk
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = SIZE_HALF
                        position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Calculate trailing stop
            current_stop = close[i] + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = entry_price - close[i]
                    risk = 2.5 * atr[i]  # initial risk
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = -SIZE_HALF
                        position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals