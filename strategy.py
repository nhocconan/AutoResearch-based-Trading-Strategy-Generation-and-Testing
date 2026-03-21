#!/usr/bin/env python3
"""
Experiment #462: 1d Daily Trend + Weekly Bias + Choppiness Regime + RSI
Hypothesis: Daily timeframe captures major trends while weekly HTF provides 
macro bias. Choppiness Index (CHOP) detects regime: CHOP>61.8 = range (mean 
revert), CHOP<38.2 = trending (trend follow). This adapts to 2022 crash and 
2025 bear market better than pure trend following. Multiple entry paths ensure 
>=10 trades requirement even on daily data.
Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_weekly_bias_rsi_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    atr = calculate_atr(high, low, close, period)
    
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = np.sum(atr[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    hma_1d = calculate_hma(close, 21)
    hma_1d_fast = calculate_hma(close, 9)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (HTF)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily HMA trend
        hma_1d_bullish = close[i] > hma_1d[i]
        hma_1d_bearish = close[i] < hma_1d[i]
        
        # Fast HMA crossover
        fast_above_slow = hma_1d_fast[i] > hma_1d[i]
        fast_below_slow = hma_1d_fast[i] < hma_1d[i]
        
        # Regime detection via Choppiness Index
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        
        # RSI levels
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Trending regime + Weekly bullish + Daily bullish + RSI pullback
        if is_trending and weekly_bullish and hma_1d_bullish and rsi[i] > 40 and rsi[i] < 55:
            new_signal = SIZE_ENTRY
        
        # Path 2: Ranging regime + Weekly bullish + RSI oversold (mean reversion)
        elif is_ranging and weekly_bullish and rsi_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 3: Fast HMA crossover up + Weekly bullish + RSI < 60
        elif fast_above_slow and weekly_bullish and rsi[i] < 60:
            new_signal = SIZE_ENTRY
        
        # Path 4: Daily bullish + RSI crossing up from oversold
        elif hma_1d_bullish and rsi_oversold and rsi[i] > rsi[i-1]:
            new_signal = SIZE_ENTRY
        
        # Path 5: Weekly bullish + Price above both HMA (trend confirmation)
        elif weekly_bullish and close[i] > hma_1d[i] and close[i] > hma_1w_aligned[i]:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Only evaluate if no long signal already set
        if new_signal == 0.0:
            # Path 1: Trending regime + Weekly bearish + Daily bearish + RSI pullback
            if is_trending and weekly_bearish and hma_1d_bearish and rsi[i] > 45 and rsi[i] < 60:
                new_signal = -SIZE_ENTRY
            
            # Path 2: Ranging regime + Weekly bearish + RSI overbought (mean reversion)
            elif is_ranging and weekly_bearish and rsi_overbought:
                new_signal = -SIZE_ENTRY
            
            # Path 3: Fast HMA crossover down + Weekly bearish + RSI > 40
            elif fast_below_slow and weekly_bearish and rsi[i] > 40:
                new_signal = -SIZE_ENTRY
            
            # Path 4: Daily bearish + RSI crossing down from overbought
            elif hma_1d_bearish and rsi_overbought and rsi[i] < rsi[i-1]:
                new_signal = -SIZE_ENTRY
            
            # Path 5: Weekly bearish + Price below both HMA (trend confirmation)
            elif weekly_bearish and close[i] < hma_1d[i] and close[i] < hma_1w_aligned[i]:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for daily timeframe)
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
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for daily timeframe)
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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