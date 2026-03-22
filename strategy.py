#!/usr/bin/env python3
"""
Experiment #491: 12h Donchian Breakout + Daily/Weekly HMA Bias + Choppiness Index Regime + ATR Stop
Hypothesis: Donchian channel breakouts capture clean trend moves while Choppiness Index filters
out whipsaw-prone ranging markets. Using BOTH Daily and Weekly HMA provides stronger HTF bias
alignment than Daily alone. 12h timeframe balances trade frequency with reduced noise vs lower TFs.
CHOP < 38.2 = trending (trade breakouts), CHOP > 61.8 = choppy (stay flat). This regime filter
should significantly improve Sharpe vs pure trend-following. Conservative sizing (0.25) with
2.0*ATR stoploss controls drawdown. Multiple entry paths ensure >=10 trades per symbol.
Timeframe: 12h (REQUIRED), HTF: 1d + 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_daily_weekly_hma_chop_regime_atr_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = np.sum(atr[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi = calculate_rsi(close, 14)
    
    # 12h HMA for additional trend confirmation
    hma_12h = calculate_hma(close, 21)
    hma_12h_fast = calculate_hma(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (HTF)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Weekly trend bias (HTF - stronger filter)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # 12h HMA trend
        hma_12h_bullish = close[i] > hma_12h[i]
        hma_12h_bearish = close[i] < hma_12h[i]
        
        # Fast HMA crossover
        fast_above_slow = hma_12h_fast[i] > hma_12h[i]
        fast_below_slow = hma_12h_fast[i] < hma_12h[i]
        
        # Choppiness regime filter
        trending_market = chop[i] < 38.2  # Good for breakouts
        choppy_market = chop[i] > 61.8   # Avoid trading
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] if i > 0 else False
        
        # Donchian position (price above/below channel)
        above_channel = close[i] > donchian_upper[i]
        below_channel = close[i] < donchian_lower[i]
        
        # RSI zones
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Donchian breakout + Daily bullish + Trending market
        if breakout_long and daily_bullish and trending_market:
            new_signal = SIZE_ENTRY
        
        # Path 2: Donchian breakout + Weekly bullish + RSI bullish
        elif breakout_long and weekly_bullish and rsi_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 3: Above channel + Daily bullish + Weekly bullish + Fast HMA crossover
        elif above_channel and daily_bullish and weekly_bullish and fast_above_slow:
            new_signal = SIZE_ENTRY
        
        # Path 4: Donchian breakout + 12h HMA bullish + RSI not overbought
        elif breakout_long and hma_12h_bullish and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        
        # Path 5: Daily bullish + Weekly bullish + Fast HMA crossover + Trending
        elif daily_bullish and weekly_bullish and fast_above_slow and trending_market:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Donchian breakout + Daily bearish + Trending market
        if breakout_short and daily_bearish and trending_market:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Donchian breakout + Weekly bearish + RSI bearish
        elif breakout_short and weekly_bearish and rsi_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Below channel + Daily bearish + Weekly bearish + Fast HMA crossover
        elif below_channel and daily_bearish and weekly_bearish and fast_below_slow:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Donchian breakout + 12h HMA bearish + RSI not oversold
        elif breakout_short and hma_12h_bearish and rsi[i] > 30:
            new_signal = -SIZE_ENTRY
        
        # Path 5: Daily bearish + Weekly bearish + Fast HMA crossover + Trending
        elif daily_bearish and weekly_bearish and fast_below_slow and trending_market:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 12h timeframe)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 12h timeframe)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
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