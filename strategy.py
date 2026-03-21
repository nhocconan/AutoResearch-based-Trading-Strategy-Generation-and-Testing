#!/usr/bin/env python3
"""
Experiment #053: 12h Fisher Transform with Daily HMA + Choppiness Regime Filter
Hypothesis: Fisher Transform excels at catching reversals in bear/range markets (2022 crash, 2025 bear).
Combine with Daily HMA(21) for trend bias and Choppiness Index(14) to avoid mean-reversion in strong trends.
12h timeframe balances trade frequency vs noise. Entry on Fisher extremes (-1.5/+1.5) aligned with HTF trend.
Position sizing: 0.30 entry, 0.15 at 2R profit, stoploss at 2.5*ATR. Fewer but higher quality trades.
This differs from #047 by using Fisher instead of Supertrend for entry timing + Choppiness regime filter.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_daily_hma_chop_regime_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Excellent for catching reversals at extremes.
    Long when Fisher crosses above -1.5 from below.
    Short when Fisher crosses below +1.5 from above.
    """
    hl2 = (high + low) / 2
    # Normalize price to -1 to +1 range
    highest = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = np.where(range_val < 1e-10, 1e-10, range_val)
    
    normalized = (hl2 - lowest) / range_val
    normalized = np.clip(normalized, 0.001, 0.999)  # Avoid log(0) or log(1)
    
    # Fisher transform
    fisher_input = 0.5 * np.log((1 + normalized) / (1 - normalized))
    fisher = pd.Series(fisher_input).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return fisher

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = range-bound (mean reversion favorable)
    CHOP < 38.2 = trending (trend following favorable)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    range_val = highest - lowest
    range_val = np.where(range_val < 1e-10, 1e-10, range_val)
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    chop = 100 * np.log10(atr_sum / range_val) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    
    return chop

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
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    fisher = calculate_fisher_transform(high, low, close, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    # 12h HMA for local trend confirmation
    hma_21 = calculate_hma(close, 21)
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
        # Daily trend filter (HTF) - use HMA slope and price position
        daily_trend_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_trend_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # Choppiness regime filter
        # CHOP < 45 = trending (favor trend-following entries)
        # CHOP > 55 = ranging (favor mean-reversion entries)
        is_trending = chop[i] < 45
        is_ranging = chop[i] > 55
        
        # 12h HMA trend confirmation
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # Fisher Transform signals
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long_signal = (i > 0) and (fisher[i] > -1.5) and (fisher[i-1] <= -1.5)
        fisher_short_signal = (i > 0) and (fisher[i] < 1.5) and (fisher[i-1] >= 1.5)
        
        # Fisher extreme levels (stronger signal)
        fisher_extreme_long = fisher[i] < -1.8
        fisher_extreme_short = fisher[i] > 1.8
        
        # RSI confirmation
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_rising = (i > 2) and (rsi[i] > rsi[i-2])
        rsi_falling = (i > 2) and (rsi[i] < rsi[i-2])
        
        new_signal = 0.0
        
        # LONG ENTRY conditions (multiple pathways for flexibility)
        # Path 1: Fisher reversal + Daily bullish trend + Trending regime
        if fisher_long_signal and daily_trend_bullish and is_trending:
            new_signal = SIZE_ENTRY
        # Path 2: Fisher extreme + Daily bullish + RSI oversold (stronger signal)
        elif fisher_extreme_long and daily_trend_bullish and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Path 3: HMA crossover + Daily bullish + RSI rising (trend continuation)
        elif hma_trend_long and daily_trend_bullish and rsi_rising and not fisher_short_signal:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        # Path 1: Fisher reversal + Daily bearish trend + Trending regime
        if fisher_short_signal and daily_trend_bearish and is_trending:
            new_signal = -SIZE_ENTRY
        # Path 2: Fisher extreme + Daily bearish + RSI overbought (stronger signal)
        elif fisher_extreme_short and daily_trend_bearish and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Path 3: HMA crossover + Daily bearish + RSI falling (trend continuation)
        elif hma_trend_short and daily_trend_bearish and rsi_falling and not fisher_long_signal:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Calculate trailing stop (2.5*ATR for 12h timeframe)
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
                    risk = 2.5 * atr[i]  # Initial risk at entry
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
                    risk = 2.5 * atr[i]  # Initial risk at entry
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