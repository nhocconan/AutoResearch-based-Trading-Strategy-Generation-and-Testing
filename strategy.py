#!/usr/bin/env python3
"""
Experiment #140: 30m Fisher Transform Reversals with 4h HMA Trend Filter
Hypothesis: 30m timeframe captures intraday swings better than 1h/4h but needs
strong HTF filter to avoid noise. Fisher Transform excels at catching reversals
in bear/range markets (2022, 2025) where trend-following fails. Combined with
4h HMA for major trend direction and RSI for momentum confirmation. This should
work in both trending (2021) and ranging/bear (2022, 2025) regimes.
Position sizing: 0.25 entry, reduce to 0.12 at 2R profit, stoploss at 2.5*ATR.
Timeframe: 30m for more responsive entries than 1h/4h but less noise than 15m.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_4h_hma_rsi_reversal_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5 from below.
    Short when Fisher crosses below +1.5 from above.
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    # Calculate median price
    median = (high + low) / 2
    
    # Normalize price to range -1 to +1
    hh = pd.Series(median).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(median).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_val = hh - ll
    range_val = np.where(range_val < 0.001, 0.001, range_val)
    
    normalized = 2 * (median - ll) / range_val - 1
    
    # Apply Fisher transform
    for i in range(period, n):
        # Smooth the normalized value
        smooth = 0.67 * normalized[i] + 0.33 * (0.67 * normalized[i-1] + 0.33 * normalized[i-2] if i > 1 else normalized[i-1])
        smooth = np.clip(smooth, -0.999, 0.999)
        
        # Fisher transform formula
        fisher[i] = 0.5 * np.log((1 + smooth) / (1 - smooth))
        
        # Trigger line (1-period lag of fisher)
        trigger[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, trigger

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower_band[0]
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            # Check if trend should flip
            if direction[i] == 1 and close[i] < lower_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            elif direction[i] == -1 and close[i] > upper_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
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
    fisher, trigger = calculate_fisher_transform(high, low, close, 9)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # 30m HMA for short-term trend
    hma_fast = calculate_hma(close, 8)
    hma_slow = calculate_hma(close, 21)
    
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
        # 4h trend filter (major trend direction)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # Fisher Transform reversal signals
        fisher_long = fisher[i] > -1.5 and trigger[i] <= -1.5  # Cross above -1.5
        fisher_short = fisher[i] < 1.5 and trigger[i] >= 1.5   # Cross below +1.5
        
        # Fisher extreme levels (stronger signals)
        fisher_oversold = fisher[i] < -2.0
        fisher_overbought = fisher[i] > 2.0
        
        # RSI confirmation
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_rising = rsi[i] > rsi[i-3] if i > 3 else False
        rsi_falling = rsi[i] < rsi[i-3] if i > 3 else False
        
        # HMA crossover
        hma_cross_long = hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]
        hma_cross_short = hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]
        
        # Volume confirmation
        volume_confirmed = vol_ratio[i] > 0.8  # At least 80% of avg volume
        
        new_signal = 0.0
        
        # LONG ENTRY: Fisher reversal + 4h bullish + RSI confirmation + Volume
        if fisher_long or fisher_oversold:
            if hma_4h_bullish and rsi_oversold and volume_confirmed:
                new_signal = SIZE_ENTRY
            elif st_bullish and rsi_rising and volume_confirmed:
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: Fisher reversal + 4h bearish + RSI confirmation + Volume
        elif fisher_short or fisher_overbought:
            if hma_4h_bearish and rsi_overbought and volume_confirmed:
                new_signal = -SIZE_ENTRY
            elif st_bearish and rsi_falling and volume_confirmed:
                new_signal = -SIZE_ENTRY
        
        # HMA crossover entry (secondary signal)
        if new_signal == 0.0:
            if hma_cross_long and hma_4h_bullish and rsi[i] > 45:
                new_signal = SIZE_ENTRY
            elif hma_cross_short and hma_4h_bearish and rsi[i] < 55:
                new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
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