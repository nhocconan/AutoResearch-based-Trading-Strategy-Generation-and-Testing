#!/usr/bin/env python3
"""
Experiment #418: 4h Fisher Transform + Daily HMA Bias + Choppiness Regime + RSI Momentum
Hypothesis: 4h timeframe captures medium-term swings better than 1d while avoiding 1h noise.
Fisher Transform identifies reversal points, Daily HMA provides trend bias, Choppiness Index
filters regime (avoid trend entries in choppy markets). Multiple entry paths ensure >=10 trades.
Key changes from #408: Faster 4h timeframe for more trades, Choppiness regime filter to avoid
whipsaws, relaxed entry conditions with 5+ paths for longs and shorts. Target: Beat Sharpe=0.499.
Timeframe: 4h (REQUIRED), HTF: 1d for trend bias via mtf_data helper.
Position size: 0.25 discrete, stoploss 2.5*ATR for 4h timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_daily_hma_chop_regime_rsi_atr_v1"
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

def calculate_fisher(close, high, low, period=9):
    """Calculate Ehlers Fisher Transform using high/low for better normalization."""
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    smoothed_prev = 0.0
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0.0
            if i > period:
                trigger[i] = fisher[i-1]
            continue
        
        normalized = 2.0 * (close[i] - lowest) / (highest - lowest) - 1.0
        normalized = np.clip(normalized, -0.99, 0.99)
        
        if i == period:
            smoothed = normalized
        else:
            smoothed = 0.67 * normalized + 0.33 * smoothed_prev
        
        smoothed_prev = smoothed
        
        fisher[i] = 0.5 * np.log((1.0 + smoothed) / (1.0 - smoothed))
        
        if i > period:
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market (mean reversion preferred)
    CHOP < 38.2 = trending market (trend following preferred)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            chop[i] = 50.0
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr_sum += max(tr1, tr2, tr3)
        
        if tr_sum > 0:
            chop[i] = 100 * np.log10((highest - lowest) / tr_sum) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
    fisher, trigger = calculate_fisher(close, high, low, 9)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    sma50 = calculate_sma(close, 50)
    sma200 = calculate_sma(close, 200)
    
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
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma50[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (long-term direction) - SOFT filter
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Choppiness regime filter
        choppy_market = chop[i] > 55.0  # Range-bound
        trending_market = chop[i] < 45.0  # Trending
        
        # Fisher Transform signals (reversal detection)
        fisher_bull_cross = fisher[i] > -1.5 and trigger[i] <= -1.5
        fisher_bear_cross = fisher[i] < 1.5 and trigger[i] >= 1.5
        
        # Fisher extreme levels
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # Fisher turning
        fisher_turning_up = fisher[i] > fisher[i-1] if i > 0 else False
        fisher_turning_down = fisher[i] < fisher[i-1] if i > 0 else False
        
        # RSI momentum (RELAXED to ensure trades)
        rsi_ok_long = rsi[i] > 25 and rsi[i] < 85
        rsi_ok_short = rsi[i] > 15 and rsi[i] < 75
        
        # Price position
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        above_sma200 = close[i] > sma200[i] if not np.isnan(sma200[i]) else False
        below_sma200 = close[i] < sma200[i] if not np.isnan(sma200[i]) else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (5+ paths to ensure >=10 trades) ===
        # Path 1: Fisher cross + Daily bullish + RSI ok (primary trend follow)
        if fisher_bull_cross and daily_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Path 2: Fisher oversold + turning up + above SMA50 (pullback entry)
        elif fisher_oversold and fisher_turning_up and above_sma50 and rsi[i] > 30:
            new_signal = SIZE_ENTRY
        # Path 3: Daily bullish + Fisher turning up + RSI momentum
        elif daily_bullish and fisher_turning_up and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Path 4: Fisher cross + above SMA50 (trend confirmation)
        elif fisher_bull_cross and above_sma50 and rsi[i] > 35:
            new_signal = SIZE_ENTRY
        # Path 5: Choppy market + Fisher oversold (mean reversion)
        elif choppy_market and fisher_oversold and rsi[i] > 25:
            new_signal = SIZE_ENTRY
        # Path 6: Trending market + Daily bullish + Fisher > -1 (trend continuation)
        elif trending_market and daily_bullish and fisher[i] > -1.0 and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (5+ paths to ensure >=10 trades) ===
        # Path 1: Fisher cross + Daily bearish + RSI ok (primary trend follow)
        if fisher_bear_cross and daily_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Path 2: Fisher overbought + turning down + below SMA50 (pullback entry)
        elif fisher_overbought and fisher_turning_down and below_sma50 and rsi[i] < 70:
            new_signal = -SIZE_ENTRY
        # Path 3: Daily bearish + Fisher turning down + RSI momentum
        elif daily_bearish and fisher_turning_down and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 4: Fisher cross + below SMA50 (trend confirmation)
        elif fisher_bear_cross and below_sma50 and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Path 5: Choppy market + Fisher overbought (mean reversion)
        elif choppy_market and fisher_overbought and rsi[i] < 75:
            new_signal = -SIZE_ENTRY
        # Path 6: Trending market + Daily bearish + Fisher < 1 (trend continuation)
        elif trending_market and daily_bearish and fisher[i] < 1.0 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals