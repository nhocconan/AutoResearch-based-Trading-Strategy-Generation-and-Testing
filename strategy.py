#!/usr/bin/env python3
"""
Experiment #155: 12h Regime-Adaptive Supertrend with Daily HMA Filter
Hypothesis: 12h timeframe captures medium-term swings better than 4h (less noise) 
while being more responsive than 1d. Combining Supertrend for trend direction,
RSI for pullback entries, and Bollinger Band Width for regime detection allows
adaptive logic: trend-follow in wide BB regimes, mean-revert in narrow BB regimes.
Daily HMA provides major trend bias. Entry thresholds loosened (RSI 35/65, not 30/70)
to ensure sufficient trades. ATR stoploss at 2.5*ATR. Position sizing: 0.25 entry,
0.15 at 2R profit. This targets both trending and ranging market conditions.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_supertrend_daily_hma_rsi_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_line, supertrend_direction (1=bullish, -1=bearish)
    """
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    n = len(close)
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish (price above supertrend)
    
    supertrend[0] = upper_band[0]
    
    for i in range(1, n):
        if direction[i-1] == 1:
            if close[i] < supertrend[i-1]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
                supertrend[i] = max(supertrend[i-1], lower_band[i])
        else:
            if close[i] > supertrend[i-1]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend[i] = min(supertrend[i-1], upper_band[i])
    
    return supertrend, direction

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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    band_width = (upper - lower) / sma
    band_width = np.where(sma > 0, band_width, 0.0)
    return upper, lower, sma, band_width

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

def calculate_bbwidth_percentile(band_width, lookback=100):
    """Calculate rolling percentile of band width for regime detection."""
    bw_s = pd.Series(band_width)
    # Calculate percentile rank within rolling window
    bw_percentile = bw_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10),
        raw=False
    ).values
    return bw_percentile

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
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend_line, supertrend_dir = calculate_supertrend(high, low, close, 10, 3.0)
    bb_upper, bb_lower, bb_mid, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    bb_percentile = calculate_bbwidth_percentile(bb_width, 100)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Daily trend filter (major trend direction)
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # 12h trend filter
        trend_bullish = hma_20[i] > hma_50[i]
        trend_bearish = hma_20[i] < hma_50[i]
        
        # Supertrend direction
        st_bullish = supertrend_dir[i] == 1
        st_bearish = supertrend_dir[i] == -1
        
        # Regime detection via BB Width percentile
        # >0.7 = wide bands (trending), <0.3 = narrow bands (ranging)
        regime_trending = bb_percentile[i] > 0.5
        regime_ranging = bb_percentile[i] < 0.5
        
        # RSI signals (loosened thresholds for more trades)
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        rsi_rising = rsi[i] > rsi[i-2] if i > 2 else False
        rsi_falling = rsi[i] < rsi[i-2] if i > 2 else False
        rsi_neutral = 40 <= rsi[i] <= 60
        
        new_signal = 0.0
        
        # TRENDING REGIME: Follow Supertrend with RSI pullback confirmation
        if regime_trending:
            # Long: Supertrend bullish + RSI pullback (not overbought) + Daily not bearish
            if st_bullish and rsi_oversold and not daily_bearish:
                new_signal = SIZE_ENTRY
            elif st_bullish and rsi_rising and trend_bullish:
                new_signal = SIZE_ENTRY
            elif st_bullish and daily_bullish and rsi_neutral:
                new_signal = SIZE_ENTRY
            
            # Short: Supertrend bearish + RSI pullback (not oversold) + Daily not bullish
            elif st_bearish and rsi_overbought and not daily_bullish:
                new_signal = -SIZE_ENTRY
            elif st_bearish and rsi_falling and trend_bearish:
                new_signal = -SIZE_ENTRY
            elif st_bearish and daily_bearish and rsi_neutral:
                new_signal = -SIZE_ENTRY
        
        # RANGING REGIME: Mean reversion with BB boundaries
        else:
            # Long: Price near BB lower + RSI oversold + Daily not strongly bearish
            if close[i] < bb_lower[i] * 0.99 and rsi_oversold and not daily_bearish:
                new_signal = SIZE_ENTRY
            elif close[i] < bb_mid[i] * 0.98 and rsi[i] < 40 and rsi_rising:
                new_signal = SIZE_ENTRY
            
            # Short: Price near BB upper + RSI overbought + Daily not strongly bullish
            elif close[i] > bb_upper[i] * 1.01 and rsi_overbought and not daily_bullish:
                new_signal = -SIZE_ENTRY
            elif close[i] > bb_mid[i] * 1.02 and rsi[i] > 60 and rsi_falling:
                new_signal = -SIZE_ENTRY
        
        # Supertrend reversal signals (works in both regimes)
        if new_signal == 0.0:
            # Supertrend flipped bullish
            if st_bullish and supertrend_dir[i-1] == -1:
                if daily_bullish or trend_bullish:
                    new_signal = SIZE_ENTRY
                elif rsi_rising:
                    new_signal = SIZE_ENTRY * 0.6
            
            # Supertrend flipped bearish
            elif st_bearish and supertrend_dir[i-1] == 1:
                if daily_bearish or trend_bearish:
                    new_signal = -SIZE_ENTRY
                elif rsi_falling:
                    new_signal = -SIZE_ENTRY * 0.6
        
        # HMA crossover signals (additional trend confirmation)
        if new_signal == 0.0:
            if trend_bullish and hma_20[i-1] <= hma_50[i-1]:
                if st_bullish or daily_bullish:
                    new_signal = SIZE_ENTRY
            elif trend_bearish and hma_20[i-1] >= hma_50[i-1]:
                if st_bearish or daily_bearish:
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