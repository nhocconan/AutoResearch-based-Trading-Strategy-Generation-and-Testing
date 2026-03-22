#!/usr/bin/env python3
"""
Experiment #042: 1d Fisher Transform + 1w HMA Trend + Choppiness Regime
Hypothesis: On daily timeframe, Ehlers Fisher Transform catches major reversals with high accuracy.
Combined with weekly HMA for major trend filter and Choppiness Index to avoid range whipsaw.
Daily timeframe = fewer but higher quality signals. Weekly HTF provides macro trend context.
Position sizing: 0.25 base, 0.35 max, discrete levels to minimize fee churn.
Stoploss: 2.5*ATR trailing stop to survive crypto volatility.
Key innovation: Fisher Transform normalized to [-1, +1] with clear reversal signals at extremes.
Choppiness Index > 61.8 = range (avoid trend trades), < 38.2 = trending (favor breakouts).
Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_1w_hma_chop_regime_v1"
timeframe = "1d"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to [-1, +1] Gaussian distribution.
    Catches reversals when Fisher crosses extreme levels.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: 0.66 * ((price - lowest) / (highest - lowest) - 0.5)
    3. Fisher: 0.5 * ln((1 + normalized) / (1 - normalized))
    4. Signal line: 1-period lag of Fisher
    
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    # Typical price
    typical = (high + low) / 2.0
    
    for i in range(period, n):
        # Highest high and lowest low over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        # Avoid division by zero
        if highest == lowest:
            fisher[i] = 0.0
        else:
            # Normalize to [-0.99, +0.99] to avoid ln(0)
            normalized = 0.66 * ((typical[i] - lowest) / (highest - lowest) - 0.5)
            normalized = np.clip(normalized, -0.99, 0.99)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Signal line (1-period lag)
        if i > 0 and not np.isnan(fisher[i - 1]):
            fisher_signal[i] = fisher[i - 1]
    
    return fisher, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    Range: 0-100
    CHOP > 61.8 = choppy/ranging market (favor mean reversion)
    CHOP < 38.2 = trending market (favor trend following)
    
    Formula:
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        
        # Highest high and lowest low over period
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI calculations
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    chop = calculate_choppiness_index(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    # Additional trend filters
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume trend (optional confirmation)
    volume = prices["volume"].values
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.35
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # 1w trend bias (HTF) - major trend filter
        bull_trend = close[i] > hma_1w_aligned[i]
        bear_trend = close[i] < hma_1w_aligned[i]
        
        # Choppiness regime
        range_regime = chop[i] > 55  # Choppy/ranging
        trend_regime = chop[i] < 45  # Trending
        
        # ADX trend strength
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20
        
        # Fisher Transform signals
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher[i] > fisher_signal[i] and fisher_signal[i] < -1.0
        fisher_cross_down = fisher[i] < fisher_signal[i] and fisher_signal[i] > 1.0
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and close[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and close[i] < ema_200[i]
        
        # Volume confirmation
        vol_above_avg = volume[i] > vol_sma[i] * 1.2 if not np.isnan(vol_sma[i]) else True
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Fisher cross up from oversold + 1w bull trend + trending regime
        if fisher_cross_up and bull_trend and trend_regime:
            new_signal = SIZE_MAX
        # Secondary: Fisher oversold + 1w bull trend + range regime (mean reversion)
        elif fisher_oversold and bull_trend and range_regime:
            new_signal = SIZE_BASE
        # Tertiary: Fisher cross up + EMA bullish + volume confirmation
        elif fisher_cross_up and ema_bullish and vol_above_avg:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: Fisher cross down from overbought + 1w bear trend + trending regime
        if fisher_cross_down and bear_trend and trend_regime:
            new_signal = -SIZE_MAX
        # Secondary: Fisher overbought + 1w bear trend + range regime (mean reversion)
        elif fisher_overbought and bear_trend and range_regime:
            new_signal = -SIZE_BASE
        # Tertiary: Fisher cross down + EMA bearish + volume confirmation
        elif fisher_cross_down and ema_bearish and vol_above_avg:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals