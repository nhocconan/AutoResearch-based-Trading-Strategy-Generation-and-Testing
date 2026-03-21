#!/usr/bin/env python3
"""
Experiment #147: 1h Regime-Adaptive Strategy with Choppiness Index + 4h HMA Trend Filter
Hypothesis: 1h timeframe needs regime detection to avoid whipsaws. Choppiness Index (CHOP)
distinguishes trending (CHOP<38.2) vs ranging (CHOP>61.8) markets. In trending regime,
follow 4h HMA direction. In ranging regime, mean-revert on RSI extremes. This adapts
to both bull (2021) and bear/range (2022, 2025) markets. Volume confirmation filters
false breakouts. Conservative sizing (0.25) controls drawdown during 2022 crash.
Timeframe: 1h balances signal frequency vs noise. HTF: 4h for trend filter.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_chop_4h_hma_rsi_volume_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market, CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    for i in range(period, n):
        hh_ll = hh[i] - ll[i]
        if hh_ll > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / hh_ll) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    return vol_ratio

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower_band[0]
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            trend[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper_band[i]
            trend[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            trend[i] = trend[i-1]
            # Check if trend should flip
            if trend[i] == 1 and close[i] < lower_band[i]:
                supertrend[i] = upper_band[i]
                trend[i] = -1
            elif trend[i] == -1 and close[i] > upper_band[i]:
                supertrend[i] = lower_band[i]
                trend[i] = 1
    
    return supertrend, trend

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    
    # HMA for local trend
    hma_fast = calculate_hma(close, 9)
    hma_slow = calculate_hma(close, 27)
    
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
        # 4h trend filter (major trend direction)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # Supertrend direction
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # HMA crossover trend
        hma_trend_long = hma_fast[i] > hma_slow[i]
        hma_trend_short = hma_fast[i] < hma_slow[i]
        
        # Volume confirmation
        volume_confirmed = vol_ratio[i] > 0.8  # At least 80% of avg volume
        
        # Regime detection via Choppiness Index
        trending_regime = chop[i] < 42.0  # Slightly lenient for more trades
        ranging_regime = chop[i] > 58.0   # Slightly lenient for more trades
        
        # RSI conditions (more lenient for trade generation)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_rising = rsi[i] > rsi[i-3] if i > 3 else False
        rsi_falling = rsi[i] < rsi[i-3] if i > 3 else False
        
        new_signal = 0.0
        
        # TRENDING REGIME: Follow trend with multiple confirmations
        if trending_regime:
            # LONG: 4h HMA bullish + Supertrend bullish + HMA trend + volume
            if hma_4h_bullish and st_bullish and hma_trend_long and volume_confirmed:
                new_signal = SIZE_ENTRY
            # SHORT: 4h HMA bearish + Supertrend bearish + HMA trend + volume
            elif hma_4h_bearish and st_bearish and hma_trend_short and volume_confirmed:
                new_signal = -SIZE_ENTRY
        
        # RANGING REGIME: Mean reversion with RSI extremes
        elif ranging_regime:
            # LONG: RSI oversold + 4h HMA not strongly bearish + volume
            if rsi_oversold and not hma_4h_bearish and volume_confirmed:
                new_signal = SIZE_ENTRY
            # SHORT: RSI overbought + 4h HMA not strongly bullish + volume
            elif rsi_overbought and not hma_4h_bullish and volume_confirmed:
                new_signal = -SIZE_ENTRY
        
        # NEUTRAL REGIME: Supertrend crossover with RSI confirmation
        else:
            # LONG: Supertrend flips bullish + RSI rising
            if st_bullish and (not st_bearish or st_trend[i-1] == -1) and rsi_rising:
                new_signal = SIZE_ENTRY
            # SHORT: Supertrend flips bearish + RSI falling
            elif st_bearish and (not st_bullish or st_trend[i-1] == 1) and rsi_falling:
                new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
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
            
            # Calculate trailing stop (2.0*ATR from lowest)
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