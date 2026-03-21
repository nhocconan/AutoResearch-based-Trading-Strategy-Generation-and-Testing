#!/usr/bin/env python3
"""
Experiment #104: 30m Regime-Adaptive Strategy with 4h HMA Trend Filter
Hypothesis: Most failed strategies use单一 approach (only trend or only mean reversion).
This strategy adapts to market regime using Choppiness Index (CHOP):
- CHOP > 61.8: Range market → Mean reversion (RSI extremes + Bollinger)
- CHOP < 38.2: Trend market → Trend following (EMA crossover + RSI pullback)
- 38.2 <= CHOP <= 61.8: Transition → Reduce position size

Use 4h HMA for HTF trend bias (proven in successful strategies).
30m timeframe provides good trade frequency while filtering 15m noise.
Position sizing: 0.25 entry, 0.15 half at profit, stoploss at 2.5*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_adaptive_4h_hma_chop_v1"
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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    return sma.values, upper.values, lower.values, bandwidth.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = Range/Consolidation
    CHOP < 38.2 = Trending
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)  # Avoid div by zero
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    trend = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper_band[0]
    
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            trend[i] = 1
        else:
            supertrend[i] = upper_band[i]
            trend[i] = -1
    
    return supertrend, trend

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
    chop = calculate_choppiness(high, low, close, 14)
    
    # EMA for trend
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # Bollinger Bands
    bb_sma, bb_upper, bb_lower, bb_bw = calculate_bollinger(close, 20, 2.0)
    
    # Supertrend
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.15
    SIZE_REDUCED = 0.10
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # 4h trend filter (HTF)
        hma_4h_valid = not np.isnan(hma_4h_aligned[i])
        daily_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        daily_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # Regime detection via Choppiness Index
        is_range = chop[i] > 61.8
        is_trend = chop[i] < 38.2
        is_transition = not is_range and not is_trend
        
        # EMA trend state
        ema_trend_long = ema_21[i] > ema_50[i]
        ema_trend_short = ema_21[i] < ema_50[i]
        
        # RSI levels
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_extreme_low = rsi[i] < 25
        rsi_extreme_high = rsi[i] > 75
        
        # Bollinger position
        bb_lower_touch = close[i] <= bb_lower[i]
        bb_upper_touch = close[i] >= bb_upper[i]
        
        # Supertrend
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        new_signal = 0.0
        
        # ===== TRENDING REGIME (CHOP < 38.2) =====
        if is_trend:
            # LONG: EMA trend + 4h bullish + Supertrend bullish + RSI pullback
            if ema_trend_long and daily_bullish and st_bullish and rsi_oversold:
                new_signal = SIZE_ENTRY
            # LONG: EMA cross + 4h bullish + Supertrend flip
            elif ema_21[i] > ema_50[i] and (i > 0 and ema_21[i-1] <= ema_50[i-1]) and daily_bullish and st_bullish:
                new_signal = SIZE_ENTRY
            
            # SHORT: EMA trend + 4h bearish + Supertrend bearish + RSI pullback
            if ema_trend_short and daily_bearish and st_bearish and rsi_overbought:
                new_signal = -SIZE_ENTRY
            # SHORT: EMA cross + 4h bearish + Supertrend flip
            elif ema_21[i] < ema_50[i] and (i > 0 and ema_21[i-1] >= ema_50[i-1]) and daily_bearish and st_bearish:
                new_signal = -SIZE_ENTRY
        
        # ===== RANGING REGIME (CHOP > 61.8) =====
        elif is_range:
            # LONG: RSI extreme oversold + BB lower touch + 4h not strongly bearish
            if rsi_extreme_low and bb_lower_touch and not daily_bearish:
                new_signal = SIZE_ENTRY
            # LONG: RSI oversold + BB lower + mean reversion
            elif rsi_oversold and bb_lower_touch:
                new_signal = SIZE_ENTRY * 0.8  # Smaller size in range
            
            # SHORT: RSI extreme overbought + BB upper touch + 4h not strongly bullish
            if rsi_extreme_high and bb_upper_touch and not daily_bullish:
                new_signal = -SIZE_ENTRY
            # SHORT: RSI overbought + BB upper + mean reversion
            elif rsi_overbought and bb_upper_touch:
                new_signal = -SIZE_ENTRY * 0.8
        
        # ===== TRANSITION REGIME (38.2 <= CHOP <= 61.8) =====
        elif is_transition:
            # Reduced position size, only strongest signals
            if ema_trend_long and daily_bullish and st_bullish and rsi_oversold:
                new_signal = SIZE_REDUCED
            elif ema_trend_short and daily_bearish and st_bearish and rsi_overbought:
                new_signal = -SIZE_REDUCED
        
        # ===== STOPLOSS LOGIC (Rule 6) =====
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
                # Take profit at 1.5R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 1.5 * risk:
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
                # Take profit at 1.5R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 1.5 * risk:
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