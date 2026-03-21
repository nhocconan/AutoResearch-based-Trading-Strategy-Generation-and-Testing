#!/usr/bin/env python3
"""
Experiment #070: 4h Regime-Adaptive Strategy with Daily HMA Filter
Hypothesis: Market regimes (trend vs range) require different approaches.
Use Choppiness Index (CHOP) to detect regime: CHOP>61.8=range (mean revert),
CHOP<38.2=trend (trend follow). Combine with Daily HMA for HTF bias.
In range: RSI extremes + Bollinger mean reversion. In trend: HMA crossover + ADX.
This adapts to 2022 crash (trend down) and 2025 bear/range market.
Position sizing: 0.25 entry, 0.15 at TP, stoploss at 2*ATR trailing.
4h timeframe provides good balance of signal quality and trade frequency.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_adaptive_daily_hma_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    atr = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = np.where(range_val == 0, 1e-10, range_val)
    
    chop = 100 * np.log10(atr_sum / range_val) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop[:period] = 50  # Default for initial bars
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return sma.values, upper.values, lower.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

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
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    bb_sma, bb_upper, bb_lower = calculate_bollinger(close, 20, 2.0)
    
    # HMA for trend following
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    
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
        # Daily trend filter (HTF) - price relative to Daily HMA
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Regime detection via Choppiness Index
        is_range = chop[i] > 55  # Range/choppy market
        is_trend = chop[i] < 45  # Trending market
        
        # Bollinger position
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        bb_low = bb_position < 0.15  # Near lower band
        bb_high = bb_position > 0.85  # Near upper band
        
        # HMA trend
    hma_trend_long = hma_16[i] > hma_48[i]
        hma_trend_short = hma_16[i] < hma_48[i]
        hma_cross_long = hma_16[i] > hma_48[i] and (i > 0 and hma_16[i-1] <= hma_48[i-1])
        hma_cross_short = hma_16[i] < hma_48[i] and (i > 0 and hma_16[i-1] >= hma_48[i-1])
        
        # ADX trend strength
        trend_strong = adx[i] > 20
        
        # RSI signals
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 35 <= rsi[i] <= 65
        
        # Directional Movement
        dm_long = plus_di[i] > minus_di[i]
        dm_short = minus_di[i] > plus_di[i]
        
        new_signal = 0.0
        
        # REGIME-ADAPTIVE ENTRY LOGIC
        
        # === RANGE REGIME: Mean Reversion ===
        if is_range:
            # Long: RSI oversold + near BB lower + Daily bullish bias
            if rsi_oversold and bb_low and daily_bullish:
                new_signal = SIZE_ENTRY
            # Short: RSI overbought + near BB upper + Daily bearish bias
            elif rsi_overbought and bb_high and daily_bearish:
                new_signal = -SIZE_ENTRY
            # Long: RSI oversold + Daily bullish (simpler condition)
            elif rsi_oversold and daily_bullish and rsi[i] < 30:
                new_signal = SIZE_ENTRY
            # Short: RSI overbought + Daily bearish (simpler condition)
            elif rsi_overbought and daily_bearish and rsi[i] > 70:
                new_signal = -SIZE_ENTRY
        
        # === TREND REGIME: Trend Following ===
        elif is_trend:
            # Long: HMA cross + Daily bullish + ADX strong
            if hma_cross_long and daily_bullish and trend_strong:
                new_signal = SIZE_ENTRY
            # Short: HMA cross + Daily bearish + ADX strong
            elif hma_cross_short and daily_bearish and trend_strong:
                new_signal = -SIZE_ENTRY
            # Long: HMA trend + Daily bullish + DM long + ADX moderate
            elif hma_trend_long and daily_bullish and dm_long and adx[i] > 15:
                new_signal = SIZE_ENTRY
            # Short: HMA trend + Daily bearish + DM short + ADX moderate
            elif hma_trend_short and daily_bearish and dm_short and adx[i] > 15:
                new_signal = -SIZE_ENTRY
        
        # === NEUTRAL REGIME: Mixed signals ===
        else:
            # Use simpler conditions to ensure trades
            if hma_trend_long and daily_bullish and rsi_neutral:
                new_signal = SIZE_ENTRY
            elif hma_trend_short and daily_bearish and rsi_neutral:
                new_signal = -SIZE_ENTRY
            elif rsi_oversold and daily_bullish:
                new_signal = SIZE_ENTRY
            elif rsi_overbought and daily_bearish:
                new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 1.5R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 1.5 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 1.5R
                risk = 2.0 * atr[i]
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