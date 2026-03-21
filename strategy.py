#!/usr/bin/env python3
"""
Experiment #460: 4h Regime-Adaptive Strategy with Daily Bias
Hypothesis: 4h timeframe captures medium-term moves while avoiding noise of lower TFs.
Key innovation: Regime-adaptive entries using Choppiness Index - mean reversion in ranges,
trend following in trends. Daily HMA provides HTF bias filter. Multiple entry paths ensure
>=10 trades requirement is met. Conservative sizing (0.25-0.30) controls drawdown.
Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_adaptive_daily_hma_chop_rsi_atr_v1"
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
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = period
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low
    
    chop = np.zeros(len(close))
    chop[:] = np.nan
    mask = (price_range > 0) & (atr_sum > 0)
    chop[mask] = 100 * np.log10(atr_sum[mask] / price_range[mask]) / np.log10(n)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return sma, upper, lower

def calculate_slope(values, lookback=5):
    """Calculate slope of values over lookback period."""
    n = len(values)
    slope = np.zeros(n)
    slope[:] = np.nan
    for i in range(lookback, n):
        if not np.isnan(values[i]) and not np.isnan(values[i - lookback]):
            slope[i] = (values[i] - values[i - lookback]) / lookback
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    hma_4h = calculate_hma(close, 21)
    hma_4h_fast = calculate_hma(close, 9)
    rsi = calculate_rsi(close, 14)
    rsi_fast = calculate_rsi(close, 7)
    chop = calculate_choppiness(high, low, close, 14)
    bb_sma, bb_upper, bb_lower = calculate_bollinger(close, 20, 2.0)
    hma_slope = calculate_slope(hma_4h, lookback=5)
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (HTF)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 4h HMA trend
        hma_4h_bullish = close[i] > hma_4h[i]
        hma_4h_bearish = close[i] < hma_4h[i]
        hma_rising = hma_slope[i] > 0
        hma_falling = hma_slope[i] < 0
        
        # Fast HMA crossover
        fast_above_slow = hma_4h_fast[i] > hma_4h[i]
        fast_below_slow = hma_4h_fast[i] < hma_4h[i]
        
        # Regime detection
        is_ranging = chop[i] > 55  # Mean reversion regime
        is_trending = chop[i] < 45  # Trend following regime
        
        # RSI zones
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        
        # Bollinger zones
        near_lower_bb = close[i] < bb_lower[i] * 1.005  # Within 0.5% of lower band
        near_upper_bb = close[i] > bb_upper[i] * 0.995  # Within 0.5% of upper band
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # TRENDING REGIME - Trend Following Entries
        if is_trending:
            # Path 1: Daily bullish + 4h bullish + RSI pullback + HMA rising
            if daily_bullish and hma_4h_bullish and rsi[i] > 35 and rsi[i] < 55 and hma_rising:
                new_signal = SIZE_ENTRY
            # Path 2: Daily bullish + Fast HMA above slow + RSI neutral
            elif daily_bullish and fast_above_slow and rsi_neutral:
                new_signal = SIZE_ENTRY
            # Path 3: 4h bullish + HMA rising + RSI not overbought
            elif hma_4h_bullish and hma_rising and rsi[i] < 60:
                new_signal = SIZE_ENTRY
        
        # RANGING REGIME - Mean Reversion Entries
        if is_ranging:
            # Path 4: Near lower Bollinger + RSI oversold + Daily bullish bias
            if near_lower_bb and rsi_oversold and daily_bullish:
                new_signal = SIZE_ENTRY
            # Path 5: RSI deeply oversold + Price above daily HMA
            elif rsi[i] < 30 and daily_bullish:
                new_signal = SIZE_ENTRY
            # Path 6: Price at lower BB + RSI < 40 (any daily bias for more trades)
            elif near_lower_bb and rsi[i] < 40:
                new_signal = SIZE_ENTRY
        
        # CROSS-REGIME entries (ensure trade frequency)
        # Path 7: Fast HMA crossover up + RSI rising
        if fast_above_slow and rsi_fast[i] > rsi_fast[i-1] and rsi[i] < 55:
            new_signal = SIZE_ENTRY
        # Path 8: Daily bullish + 4h HMA crossover
        if daily_bullish and fast_above_slow and hma_4h_fast[i] > hma_4h_fast[i-1]:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # TRENDING REGIME - Trend Following Entries
        if is_trending:
            # Path 1: Daily bearish + 4h bearish + RSI pullback + HMA falling
            if daily_bearish and hma_4h_bearish and rsi[i] > 45 and rsi[i] < 65 and hma_falling:
                new_signal = -SIZE_ENTRY
            # Path 2: Daily bearish + Fast HMA below slow + RSI neutral
            elif daily_bearish and fast_below_slow and rsi_neutral:
                new_signal = -SIZE_ENTRY
            # Path 3: 4h bearish + HMA falling + RSI not oversold
            elif hma_4h_bearish and hma_falling and rsi[i] > 40:
                new_signal = -SIZE_ENTRY
        
        # RANGING REGIME - Mean Reversion Entries
        if is_ranging:
            # Path 4: Near upper Bollinger + RSI overbought + Daily bearish bias
            if near_upper_bb and rsi_overbought and daily_bearish:
                new_signal = -SIZE_ENTRY
            # Path 5: RSI deeply overbought + Price below daily HMA
            elif rsi[i] > 70 and daily_bearish:
                new_signal = -SIZE_ENTRY
            # Path 6: Price at upper BB + RSI > 60 (any daily bias for more trades)
            elif near_upper_bb and rsi[i] > 60:
                new_signal = -SIZE_ENTRY
        
        # CROSS-REGIME entries (ensure trade frequency)
        # Path 7: Fast HMA crossover down + RSI falling
        if fast_below_slow and rsi_fast[i] < rsi_fast[i-1] and rsi[i] > 45:
            new_signal = -SIZE_ENTRY
        # Path 8: Daily bearish + 4h HMA crossover
        if daily_bearish and fast_below_slow and hma_4h_fast[i] < hma_4h_fast[i-1]:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
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