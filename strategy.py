#!/usr/bin/env python3
"""
Experiment #054: 1d Supertrend with Weekly HMA + Choppiness Regime Filter
Hypothesis: Daily timeframe needs regime detection to avoid whipsaws in 2022 crash.
Use Weekly HMA for major trend bias, Daily Supertrend for entry signals,
and Choppiness Index to distinguish trending vs ranging markets.
In trending regimes (CHOP<38.2): follow Supertrend direction with Weekly HMA confirmation.
In ranging regimes (CHOP>61.8): mean revert using RSI extremes with SMA200 filter.
This adapts to market conditions instead of forcing one style.
Position sizing: 0.25 entry, 0.125 at 2R profit, 2*ATR trailing stop.
Timeframe: 1d (required for this experiment), HTF: 1w for trend bias.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_supertrend_weekly_hma_chop_regime_v1"
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
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))
    
    supertrend[0] = lower[0]
    direction[0] = 1
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
    
    return supertrend, direction

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Price range
    price_range = highest_high - lowest_low
    
    # Choppiness calculation
    chop = np.zeros(len(close))
    mask = (price_range > 0) & (atr_sum > 0)
    chop[mask] = 100 * np.log10(atr_sum[mask] / price_range[mask]) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    chop = calculate_choppiness(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
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
    
    for i in range(250, n):  # Start after 250 bars for SMA200 warmup
        # Weekly trend filter (HTF)
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # Daily Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip signals
        st_flip_long = (i > 0) and (st_direction[i] == 1) and (st_direction[i-1] == -1)
        st_flip_short = (i > 0) and (st_direction[i] == -1) and (st_direction[i-1] == 1)
        
        # Daily HMA trend
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # Price vs SMA200
        above_sma200 = sma_200[i] > 0 and close[i] > sma_200[i]
        below_sma200 = sma_200[i] > 0 and close[i] < sma_200[i]
        
        # Regime detection via Choppiness
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        
        # RSI signals for mean reversion
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_rising = (i > 2) and (rsi[i] > rsi[i-2])
        rsi_falling = (i > 2) and (rsi[i] < rsi[i-2])
        
        new_signal = 0.0
        
        # TRENDING REGIME: Follow Supertrend with Weekly HMA confirmation
        if trending_regime:
            # Long: Supertrend flip + Weekly bullish + Daily HMA bullish
            if st_flip_long and weekly_bullish and hma_trend_long:
                new_signal = SIZE_ENTRY
            # Short: Supertrend flip + Weekly bearish + Daily HMA bearish
            elif st_flip_short and weekly_bearish and hma_trend_short:
                new_signal = -SIZE_ENTRY
            # Continuation entries in strong trend
            elif st_long and weekly_bullish and above_sma200 and rsi_rising and rsi[i] < 60:
                new_signal = SIZE_ENTRY
            elif st_short and weekly_bearish and below_sma200 and rsi_falling and rsi[i] > 40:
                new_signal = -SIZE_ENTRY
        
        # RANGING REGIME: Mean reversion with RSI extremes
        elif ranging_regime:
            # Long: RSI oversold + Price above SMA200 (bullish range)
            if rsi_oversold and above_sma200 and rsi_rising:
                new_signal = SIZE_ENTRY
            # Short: RSI overbought + Price below SMA200 (bearish range)
            elif rsi_overbought and below_sma200 and rsi_falling:
                new_signal = -SIZE_ENTRY
        
        # NEUTRAL REGIME (38.2 <= CHOP <= 61.8): Conservative trend following
        else:
            # Only enter on clear Supertrend flips with Weekly confirmation
            if st_flip_long and weekly_bullish:
                new_signal = SIZE_ENTRY
            elif st_flip_short and weekly_bearish:
                new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Calculate trailing stop
            current_stop = close[i] - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = close[i] - entry_price
                    risk = 2.0 * atr[i]  # Initial risk at entry
                    if profit >= 2.0 * risk:
                        new_signal = SIZE_HALF
                        position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Calculate trailing stop
            current_stop = close[i] + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = entry_price - close[i]
                    risk = 2.0 * atr[i]  # Initial risk at entry
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
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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