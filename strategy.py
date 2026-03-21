#!/usr/bin/env python3
"""
Experiment #126: 1d Fisher Transform + Weekly HMA Trend + Regime Filter
Hypothesis: Daily timeframe needs reversal-catching capability for bear markets (2025).
Fisher Transform excels at identifying turning points with normalized -1 to +1 range.
Combine with Weekly HMA for major trend filter, Choppiness Index for regime detection.
In trending regime (CHOP<38.2): follow Fisher signals with weekly trend.
In ranging regime (CHOP>61.8): fade Fisher extremes (mean reversion).
This adapts to both 2021-2024 bull/bear cycles and 2025 range market.
Position sizing: 0.30 entry, 0.15 at 2R profit, stoploss at 2.5*ATR trailing.
Timeframe: 1d naturally reduces trade frequency and fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_weekly_hma_chop_regime_v1"
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
    Calculate Ehlers Fisher Transform.
    Normalizes price to -1 to +1 range, highlights turning points.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    hl2 = (high + low) / 2
    
    # Calculate highest high and lowest low over period
    hh = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    
    # Normalize to 0-1 range
    range_val = hh - ll
    range_val = np.where(range_val == 0, 0.001, range_val)  # avoid div by zero
    norm = (hl2 - ll) / range_val
    norm = np.clip(norm, 0.001, 0.999)  # keep within bounds for log
    
    # Fisher calculation
    fisher = 0.5 * np.log((1 + norm) / (1 - norm + 0.0001))
    fisher = np.nan_to_num(fisher, nan=0.0)
    
    # Signal line (previous Fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        range_val = highest_high - lowest_low
        if range_val > 0 and tr_sum > 0:
            chop[i] = 100 * np.log10(tr_sum / range_val) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop[:period] = 50.0
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    hma_1d = calculate_hma(close, 21)
    
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
        # Weekly trend filter
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily trend (HMA slope)
        daily_trend_long = hma_1d[i] > hma_1d[i-1] if i > 0 else False
        daily_trend_short = hma_1d[i] < hma_1d[i-1] if i > 0 else False
        
        # Regime detection
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        neutral_regime = not trending_regime and not ranging_regime
        
        # Fisher signals
        fisher_bull_cross = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_bear_cross = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        fisher_extreme_long = fisher[i] < -1.8
        fisher_extreme_short = fisher[i] > 1.8
        
        # RSI confirmation
        rsi_long_ok = rsi[i] > 40
        rsi_short_ok = rsi[i] < 60
        
        new_signal = 0.0
        
        # LONG ENTRY - Multiple paths for trade frequency
        if trending_regime:
            # Trend following: Fisher cross + weekly trend + daily trend
            if fisher_bull_cross and weekly_bullish and daily_trend_long:
                new_signal = SIZE_ENTRY
            # Simpler: Fisher cross + weekly bullish
            elif fisher_bull_cross and weekly_bullish and rsi_long_ok:
                new_signal = SIZE_ENTRY
        elif ranging_regime:
            # Mean reversion: Fisher extreme + RSI confirmation
            if fisher_extreme_long and rsi[i] < 50:
                new_signal = SIZE_ENTRY
            # Fisher cross in range
            elif fisher_bull_cross and rsi[i] < 55:
                new_signal = SIZE_ENTRY
        else:
            # Neutral regime: use both weekly filter and Fisher
            if fisher_bull_cross and weekly_bullish:
                new_signal = SIZE_ENTRY
            elif fisher_extreme_long and weekly_bullish:
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY
        if trending_regime:
            if fisher_bear_cross and weekly_bearish and daily_trend_short:
                new_signal = -SIZE_ENTRY
            elif fisher_bear_cross and weekly_bearish and rsi_short_ok:
                new_signal = -SIZE_ENTRY
        elif ranging_regime:
            if fisher_extreme_short and rsi[i] > 50:
                new_signal = -SIZE_ENTRY
            elif fisher_bear_cross and rsi[i] > 45:
                new_signal = -SIZE_ENTRY
        else:
            if fisher_bear_cross and weekly_bearish:
                new_signal = -SIZE_ENTRY
            elif fisher_extreme_short and weekly_bearish:
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