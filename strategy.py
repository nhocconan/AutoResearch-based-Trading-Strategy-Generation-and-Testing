#!/usr/bin/env python3
"""
Experiment #113: 12h Regime-Adaptive Strategy with Daily HMA + Choppiness Filter
Hypothesis: Most failed strategies use one-size-fits-all logic. Market regimes
(trending vs ranging) require different approaches. Use Choppiness Index (CHOP)
to detect regime: CHOP>61.8 = range (mean revert), CHOP<38.2 = trend (follow).
Combine with Daily HMA for HTF bias. In trends: EMA+ADX entries. In ranges:
RSI extremes + Bollinger mean reversion. This should improve Sharpe by avoiding
whipsaw in ranges and catching trends efficiently. 12h TF reduces noise vs lower TFs.
Position sizing: 0.25 entry, 0.15 half at profit, 2.5*ATR trailing stop.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_adaptive_daily_hma_chop_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of ATR over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - low_s.rolling(window=period, min_periods=period).min()
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / hh_ll) / np.log10(period)
    chop = chop.fillna(50).values  # Default to neutral if NaN
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return sma.values, upper.values, lower.values

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
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    
    # EMA for trend
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # Bollinger Bands for mean reversion
    bb_sma, bb_upper, bb_lower = calculate_bollinger(close, 20, 2.0)
    
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
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        is_neutral = not is_trending and not is_ranging
        
        # 12h EMA trend state
        ema_trend_long = ema_21[i] > ema_50[i]
        ema_trend_short = ema_21[i] < ema_50[i]
        ema_cross_long = ema_trend_long and (i > 0 and ema_21[i-1] <= ema_50[i-1])
        ema_cross_short = ema_trend_short and (i > 0 and ema_21[i-1] >= ema_50[i-1])
        
        # ADX trend strength
        trend_strong = adx[i] > 20
        
        # Directional Movement
        dm_long = plus_di[i] > minus_di[i]
        dm_short = minus_di[i] > plus_di[i]
        
        # RSI for mean reversion
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_extreme_long = rsi[i] < 30
        rsi_extreme_short = rsi[i] > 70
        
        # Bollinger position
        bb_long = close[i] < bb_lower[i]
        bb_short = close[i] > bb_upper[i]
        
        new_signal = 0.0
        
        # TRENDING REGIME: Use trend-following logic
        if is_trending:
            # LONG: EMA trend + Daily bullish + ADX strong + DM long
            if ema_trend_long and daily_bullish and trend_strong and dm_long:
                new_signal = SIZE_ENTRY
            # LONG: EMA cross + Daily bullish
            elif ema_cross_long and daily_bullish:
                new_signal = SIZE_ENTRY
            
            # SHORT: EMA trend + Daily bearish + ADX strong + DM short
            if ema_trend_short and daily_bearish and trend_strong and dm_short:
                new_signal = -SIZE_ENTRY
            # SHORT: EMA cross + Daily bearish
            elif ema_cross_short and daily_bearish:
                new_signal = -SIZE_ENTRY
        
        # RANGING REGIME: Use mean reversion logic
        elif is_ranging:
            # LONG: RSI oversold + Daily bullish + Price below BB lower
            if rsi_oversold and daily_bullish and bb_long:
                new_signal = SIZE_ENTRY
            # LONG: RSI extreme + Daily bullish
            elif rsi_extreme_long and daily_bullish:
                new_signal = SIZE_ENTRY
            
            # SHORT: RSI overbought + Daily bearish + Price above BB upper
            if rsi_overbought and daily_bearish and bb_short:
                new_signal = -SIZE_ENTRY
            # SHORT: RSI extreme + Daily bearish
            elif rsi_extreme_short and daily_bearish:
                new_signal = -SIZE_ENTRY
        
        # NEUTRAL REGIME: Mixed approach (looser conditions to ensure trades)
        else:
            # LONG: EMA trend + Daily bullish + RSI not overbought
            if ema_trend_long and daily_bullish and rsi[i] < 70:
                new_signal = SIZE_ENTRY
            # LONG: RSI oversold + Daily bullish
            elif rsi_oversold and daily_bullish:
                new_signal = SIZE_ENTRY
            
            # SHORT: EMA trend + Daily bearish + RSI not oversold
            if ema_trend_short and daily_bearish and rsi[i] > 30:
                new_signal = -SIZE_ENTRY
            # SHORT: RSI overbought + Daily bearish
            elif rsi_overbought and daily_bearish:
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
                # Take profit at 1.5R
                profit = close[i] - entry_price
                risk = 2.5 * atr[i]
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
                profit = entry_price - close[i]
                risk = 2.5 * atr[i]
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