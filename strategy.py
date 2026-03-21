#!/usr/bin/env python3
"""
Experiment #130: 4h Regime-Adaptive Strategy with Daily/Weekly HMA Filter
Hypothesis: Previous 4h strategies failed because they used single approach (trend OR mean reversion).
This strategy adapts to market regime using Choppiness Index:
- CHOP > 61.8 = ranging market → use mean reversion (RSI extremes + Bollinger)
- CHOP < 38.2 = trending market → use trend following (HMA slope + ADX)
Use 1d HMA for primary trend filter, 1w HMA for meta-filter (only trade with weekly trend).
ADX > 25 confirms trend strength before trend entries (reduces whipsaw).
Bollinger Band squeeze detection for volatility breakout confirmation.
Position sizing: 0.25 entry, stoploss at 2.5*ATR trailing.
Timeframe: 4h with 1d/1w HTF filters for stronger signal quality.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_adaptive_daily_weekly_hma_chop_adx_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx, plus_di, minus_di

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
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
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / (sma + 1e-10)
    return upper, lower, sma, bandwidth

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_sma, bb_bw = calculate_bollinger_bands(close, 20, 2.0)
    zscore = calculate_zscore(close, 20)
    hma_4h = calculate_hma(close, 21)
    
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
        # HTF trend filters (Daily and Weekly HMA)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # HMA slope on 4h
        hma_slope_long = hma_4h[i] > hma_4h[i-1] if i > 0 else False
        hma_slope_short = hma_4h[i] < hma_4h[i-1] if i > 0 else False
        
        # Regime detection via Choppiness Index
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # ADX trend strength
        adx_strong = adx[i] > 25
        adx_weak = adx[i] <= 25
        
        # Bollinger Band position
        bb_lower_touch = close[i] <= bb_lower[i]
        bb_upper_touch = close[i] >= bb_upper[i]
        bb_squeeze = bb_bw[i] < bb_bw[i-1] if i > 0 else False
        
        # RSI extremes for mean reversion
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = 35 <= rsi[i] <= 65
        
        # Z-score extremes
        zscore_oversold = zscore[i] < -1.5
        zscore_overbought = zscore[i] > 1.5
        
        new_signal = 0.0
        
        # TRENDING REGIME entries (CHOP < 38.2)
        if is_trending and adx_strong:
            # LONG: Weekly bullish + Daily bullish + HMA slope up + ADX strong
            if weekly_bullish and daily_bullish and hma_slope_long and plus_di[i] > minus_di[i]:
                new_signal = SIZE_ENTRY
            # SHORT: Weekly bearish + Daily bearish + HMA slope down + ADX strong
            elif weekly_bearish and daily_bearish and hma_slope_short and minus_di[i] > plus_di[i]:
                new_signal = -SIZE_ENTRY
        
        # RANGING REGIME entries (CHOP > 61.8) - Mean Reversion
        elif is_ranging:
            # LONG: RSI oversold + BB lower touch + Weekly not strongly bearish
            if rsi_oversold and bb_lower_touch and not weekly_bearish:
                new_signal = SIZE_ENTRY
            # LONG: Z-score oversold + Daily bullish bias
            elif zscore_oversold and daily_bullish and rsi[i] < 40:
                new_signal = SIZE_ENTRY
            # SHORT: RSI overbought + BB upper touch + Weekly not strongly bullish
            elif rsi_overbought and bb_upper_touch and not weekly_bullish:
                new_signal = -SIZE_ENTRY
            # SHORT: Z-score overbought + Daily bearish bias
            elif zscore_overbought and daily_bearish and rsi[i] > 60:
                new_signal = -SIZE_ENTRY
        
        # TRANSITION REGIME (38.2 <= CHOP <= 61.8) - Conservative entries
        else:
            # LONG: Strong weekly trend + RSI pullback + HMA alignment
            if weekly_bullish and daily_bullish and rsi_neutral and hma_slope_long:
                new_signal = SIZE_ENTRY
            # SHORT: Strong weekly trend + RSI pullback + HMA alignment
            elif weekly_bearish and daily_bearish and rsi_neutral and hma_slope_short:
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