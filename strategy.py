#!/usr/bin/env python3
"""
Experiment #356: 30m Fisher Transform + Choppiness Regime + 4h HMA Trend

Hypothesis: 30m timeframe with Fisher Transform catches reversals better than RSI.
Choppiness Index (CHOP) detects regime: CHOP>55 = range (mean revert), CHOP<45 = trend.
4h HMA provides macro trend bias. Fisher crosses at extreme levels trigger entries.
This should work better in 2025 bear/range market than pure trend strategies.

Key insight from failures: 30m Supertrend/KAMA whipsawed badly. Fisher + CHOP adapts to regime.
Position sizing: 0.25 entry, 0.125 half at profit. Stoploss at 2*ATR.
Target: Beat Sharpe=0.499 with 30+ trades, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_chop_regime_4h_hma_atr_v1"
timeframe = "30m"
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

def calculate_fisher(close, period=9):
    """Calculate Ehlers Fisher Transform for reversal detection."""
    close_s = pd.Series(close)
    high = close_s.rolling(window=period, min_periods=period).max().values
    low = close_s.rolling(window=period, min_periods=period).min().values
    
    # Normalize price to 0-1 range
    hl_range = high - low
    hl_range = np.where(hl_range < 1e-10, 1e-10, hl_range)
    norm = (close - low) / hl_range
    norm = np.clip(norm * 2 - 1, -0.99, 0.99)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + norm) / (1 - norm + 1e-10))
    fisher = pd.Series(fisher).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return fisher

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP) for regime detection."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # High-Low range over period
    hl_max = pd.Series(high).rolling(window=period, min_periods=period).max().values
    hl_min = pd.Series(low).rolling(window=period, min_periods=period).min().values
    hl_range = hl_max - hl_min
    hl_range = np.where(hl_range < 1e-10, 1e-10, hl_range)
    
    # CHOP = 100 * log10(sum(ATR) / (High-Low)) / log10(period)
    chop = 100 * np.log10(atr_sum / hl_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    fisher = calculate_fisher(close, 9)
    fisher_prev = np.roll(fisher, 1)
    fisher_prev[0] = fisher[0]
    chop = calculate_choppiness(high, low, close, 14)
    
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
        if np.isnan(atr[i]) or np.isnan(fisher[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # 4h macro trend bias
        hma_4h_val = hma_4h_aligned[i]
        trend_bullish = not np.isnan(hma_4h_val) and close[i] > hma_4h_val
        trend_bearish = not np.isnan(hma_4h_val) and close[i] < hma_4h_val
        
        # Choppiness regime (LOOSE thresholds for more trades)
        chop_val = chop[i]
        is_range = chop_val > 50.0
        is_trend = chop_val < 50.0
        
        # Fisher Transform signals (LOOSE for trade frequency)
        fisher_cross_long = fisher_prev[i] < -1.0 and fisher[i] >= -1.0
        fisher_cross_short = fisher_prev[i] > 1.0 and fisher[i] <= 1.0
        
        # Fisher extreme levels
        fisher_oversold = fisher[i] < -0.5
        fisher_overbought = fisher[i] > 0.5
        
        # Fisher momentum (rising/falling)
        fisher_rising = fisher[i] > fisher_prev[i]
        fisher_falling = fisher[i] < fisher_prev[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Range market: Fisher mean reversion (oversold + cross up)
        if is_range and fisher_oversold and fisher_cross_long:
            new_signal = SIZE_ENTRY
        # Trend market: Fisher pullback in uptrend
        elif is_trend and trend_bullish and fisher_oversold and fisher_rising:
            new_signal = SIZE_ENTRY
        # General: Strong Fisher cross regardless of regime (ensures trades)
        elif fisher_cross_long:
            new_signal = SIZE_ENTRY
        # Backup: Price above 4h HMA + Fisher rising (trend continuation)
        elif trend_bullish and fisher_rising and fisher[i] > -0.5:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Range market: Fisher mean reversion (overbought + cross down)
        if is_range and fisher_overbought and fisher_cross_short:
            new_signal = -SIZE_ENTRY
        # Trend market: Fisher pullback in downtrend
        elif is_trend and trend_bearish and fisher_overbought and fisher_falling:
            new_signal = -SIZE_ENTRY
        # General: Strong Fisher cross regardless of regime (ensures trades)
        elif fisher_cross_short:
            new_signal = -SIZE_ENTRY
        # Backup: Price below 4h HMA + Fisher falling (trend continuation)
        elif trend_bearish and fisher_falling and fisher[i] < 0.5:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC ===
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                risk = 2.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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