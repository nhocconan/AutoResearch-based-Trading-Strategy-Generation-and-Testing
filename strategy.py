#!/usr/bin/env python3
"""
Experiment #171: 1h Regime-Adaptive Strategy with 4h/12h HMA Trend Filter
Hypothesis: 1h timeframe captures intraday swings while 4h/12h HMA provides 
major trend bias. Regime detection (Choppiness Index + ADX) switches between
trend-following (CHOP<45, ADX>25) and mean-reversion (CHOP>55, ADX<20).
RSI pullback entries in trends (RSI 40-60), RSI extreme entries in ranges (RSI<30/>70).
ATR stoploss at 2.5*ATR protects capital. Position sizing: 0.25 entry, 0.125 at 2R.
This targets all market regimes: 2021 bull (trend), 2022 crash (trend short), 2025 range (mean revert).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_4h_12h_hma_rsi_chop_adx_v1"
timeframe = "1h"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl > 0, range_hl, 1e-10)
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    chop = 100 * np.log10(atr_sum / (range_hl * period))
    chop = np.where(np.isnan(chop), 50.0, chop)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    trend = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower_band[0]
    
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            trend[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            trend[i] = -1
    
    return supertrend, trend

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters
        hma_4h_valid = hma_4h_aligned[i] > 0
        hma_12h_valid = hma_12h_aligned[i] > 0
        
        four_hour_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        four_hour_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        twelve_hour_bullish = hma_12h_valid and close[i] > hma_12h_aligned[i]
        twelve_hour_bearish = hma_12h_valid and close[i] < hma_12h_aligned[i]
        
        # Regime detection
        is_ranging = chop[i] > 55.0 and adx[i] < 25.0
        is_trending = chop[i] < 45.0 and adx[i] > 20.0
        
        # 1h trend
        trend_bullish = hma_20[i] > hma_50[i] and st_trend[i] == 1
        trend_bearish = hma_20[i] < hma_50[i] and st_trend[i] == -1
        
        # RSI signals
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_neutral_low = 35.0 <= rsi[i] <= 45.0
        rsi_neutral_high = 55.0 <= rsi[i] <= 65.0
        rsi_rising = rsi[i] > rsi[i-2] if i > 2 else False
        rsi_falling = rsi[i] < rsi[i-2] if i > 2 else False
        
        new_signal = 0.0
        
        # === MEAN REVERSION MODE (ranging market) ===
        if is_ranging:
            # Long: RSI oversold + price below 4h HMA support
            if rsi_oversold and rsi_rising:
                if not twelve_hour_bearish:
                    new_signal = SIZE_ENTRY
            
            # Short: RSI overbought + price above 4h HMA resistance
            elif rsi_overbought and rsi_falling:
                if not twelve_hour_bullish:
                    new_signal = -SIZE_ENTRY
        
        # === TREND FOLLOWING MODE (trending market) ===
        elif is_trending:
            # Long: Trend bullish + RSI pullback + 4h/12h bullish
            if trend_bullish and rsi_neutral_low and rsi_rising:
                if four_hour_bullish or twelve_hour_bullish:
                    new_signal = SIZE_ENTRY
            
            # Short: Trend bearish + RSI pullback + 4h/12h bearish
            elif trend_bearish and rsi_neutral_high and rsi_falling:
                if four_hour_bearish or twelve_hour_bearish:
                    new_signal = -SIZE_ENTRY
            
            # Supertrend continuation
            elif st_trend[i] == 1 and st_trend[i-1] == -1:
                if four_hour_bullish:
                    new_signal = SIZE_ENTRY
            elif st_trend[i] == -1 and st_trend[i-1] == 1:
                if four_hour_bearish:
                    new_signal = -SIZE_ENTRY
        
        # === TRANSITION MODE (unclear regime) ===
        else:
            # Only enter on strong HTF confirmation
            if trend_bullish and four_hour_bullish and twelve_hour_bullish:
                if rsi[i] < 50 and rsi_rising:
                    new_signal = SIZE_ENTRY
            elif trend_bearish and four_hour_bearish and twelve_hour_bearish:
                if rsi[i] > 50 and rsi_falling:
                    new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
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