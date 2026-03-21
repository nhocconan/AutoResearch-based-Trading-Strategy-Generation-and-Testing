#!/usr/bin/env python3
"""
Experiment #059: 12h Donchian Breakout with Daily/Weekly HMA Triple Trend Filter
Hypothesis: Donchian breakouts (20-period) provide clear trend-following entries.
Filter with 1d HMA for intermediate trend and 1w HMA for major trend bias.
Triple alignment (12h breakout + 1d trend + 1w trend) reduces whipsaw significantly.
Add volume confirmation to filter false breakouts (volume > 1.5x 20-period avg).
Use ATR trailing stop (2.5*ATR) for risk management with proper position tracking.
Position sizing: 0.25 entry, reduce to 0.125 at 2R profit, exit at stoploss.
12h timeframe balances trade frequency vs noise - should generate 20-40 trades/year.
Key improvement over #047: Donchian breakouts are cleaner than Supertrend flips,
and triple HTF filter (12h+1d+1w) provides stronger trend confirmation.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_daily_weekly_hma_volume_v1"
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
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    donchian_upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    return donchian_upper, donchian_lower, donchian_mid

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    vol_sma = calculate_volume_sma(volume, 20)
    
    # 12h HMA for local trend confirmation
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend filter (HTF) - price relative to HMA
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # Weekly trend filter (HTF) - major trend bias
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # 12h HMA trend confirmation
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # Volume confirmation (volume > 1.5x average)
        volume_confirmed = volume[i] > 1.5 * vol_sma[i] if vol_sma[i] > 0 else False
        
        # Previous bar signals for flip detection
        prev_breakout_long = close[i-1] > donchian_upper[i-2] if i > 1 else False
        prev_breakout_short = close[i-1] < donchian_lower[i-2] if i > 1 else False
        
        # Fresh breakout (not continuing from previous bar)
        fresh_breakout_long = breakout_long and not prev_breakout_long
        fresh_breakout_short = breakout_short and not prev_breakout_short
        
        new_signal = 0.0
        
        # LONG ENTRY: Donchian breakout + Daily bullish + Weekly bullish + Volume
        if fresh_breakout_long and daily_bullish and weekly_bullish:
            new_signal = SIZE_ENTRY
        elif fresh_breakout_long and daily_bullish and hma_trend_long and volume_confirmed:
            new_signal = SIZE_ENTRY
        elif fresh_breakout_long and weekly_bullish and hma_trend_long:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: Donchian breakout + Daily bearish + Weekly bearish + Volume
        if fresh_breakout_short and daily_bearish and weekly_bearish:
            new_signal = -SIZE_ENTRY
        elif fresh_breakout_short and daily_bearish and hma_trend_short and volume_confirmed:
            new_signal = -SIZE_ENTRY
        elif fresh_breakout_short and weekly_bearish and hma_trend_short:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Calculate trailing stop (2.5*ATR)
            current_stop = close[i] - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced and entry_atr > 0:
                    profit = close[i] - entry_price
                    risk = 2.5 * entry_atr
                    if profit >= 2.0 * risk:
                        new_signal = SIZE_HALF
                        position_reduced = True
            
            # Exit if Donchian lower breaks (trend reversal signal)
            if close[i] < donchian_lower[i] and new_signal != 0.0:
                new_signal = 0.0
        
        if position_side < 0 and entry_price > 0:
            # Calculate trailing stop (2.5*ATR)
            current_stop = close[i] + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced and entry_atr > 0:
                    profit = entry_price - close[i]
                    risk = 2.5 * entry_atr
                    if profit >= 2.0 * risk:
                        new_signal = -SIZE_HALF
                        position_reduced = True
            
            # Exit if Donchian upper breaks (trend reversal signal)
            if close[i] > donchian_upper[i] and new_signal != 0.0:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            position_reduced = False
            entry_atr = atr[i]
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            position_reduced = False
            entry_atr = atr[i]
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            position_reduced = False
            entry_atr = 0.0
        
        signals[i] = new_signal
    
    return signals