#!/usr/bin/env python3
"""
Experiment #221: 12h Fisher Transform + Choppiness Index Regime Filter with Daily HMA
Hypothesis: Fisher Transform catches reversals better than RSI in bear/range markets (2025 test period).
Choppiness Index determines regime: CHOP<38.2=trending (follow Fisher with trend), CHOP>61.8=ranging (fade Fisher extremes).
Daily HMA provides macro bias. This should work better than pure breakout (Donchian failed) and pure trend (Supertrend exhausted).
Position sizing: 0.25 entry, 0.12 half at 2R. Stoploss: 2.5*ATR trailing. Target: Beat Sharpe=0.499.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_chop_regime_daily_hma_atr_v1"
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
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Extreme values (>1.5 or <-1.5) indicate potential reversals.
    """
    hl2 = (high + low) / 2
    hl2_s = pd.Series(hl2)
    
    # Calculate highest high and lowest low over period
    highest = hl2_s.rolling(window=period, min_periods=period).max().values
    lowest = hl2_s.rolling(window=period, min_periods=period).min().values
    
    # Normalize to 0-1 range
    range_hl = highest - lowest
    range_hl = np.where(range_hl == 0, 0.001, range_hl)  # avoid div by zero
    normalized = (hl2 - lowest) / range_hl
    
    # Clamp to avoid extreme values
    normalized = np.clip(normalized, 0.001, 0.999)
    
    # Fisher transform
    fisher_input = 0.66 * ((normalized - 0.5) / 0.5) + 0.67 * np.roll(fisher_input_raw(normalized), 1)
    fisher_input = np.clip(fisher_input, -0.999, 0.999)
    
    # Actual Fisher calculation
    fisher = 0.5 * np.log((1 + fisher_input) / (1 - fisher_input + 0.0001))
    fisher_prev = np.roll(fisher, 1)
    fisher_prev[0] = fisher[0]
    
    # Signal line (trigger)
    trigger = np.roll(fisher, 1)
    trigger[0] = fisher[0]
    
    return fisher, trigger

def fisher_input_raw(normalized):
    """Helper for Fisher input calculation."""
    return 0.66 * ((normalized - 0.5) / 0.5)

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending.
    CHOP > 61.8 = ranging/consolidation
    CHOP < 38.2 = trending
    """
    atr = calculate_atr(high, low, close, period)
    
    # Highest high and lowest low over period
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Choppiness calculation
    range_hl = highest - lowest
    range_hl = np.where(range_hl == 0, 0.001, range_hl)
    
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
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
        # HTF trend filter
        daily_bullish = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else False
        daily_bearish = close[i] < hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else False
        
        # Regime detection via Choppiness Index
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        neutral_regime = not trending_regime and not ranging_regime
        
        # Fisher Transform signals
        fisher_bullish_cross = fisher[i] > fisher_trigger[i] and fisher[i-1] <= fisher_trigger[i-1]
        fisher_bearish_cross = fisher[i] < fisher_trigger[i] and fisher[i-1] >= fisher_trigger[i-1]
        
        # Fisher extreme levels for mean reversion
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # RSI confirmation
        rsi_bullish = rsi[i] > 45
        rsi_bearish = rsi[i] < 55
        rsi_neutral = 35 < rsi[i] < 65
        
        new_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 38.2) ===
        # Follow Fisher crosses with trend alignment
        if trending_regime:
            # Long: Fisher bullish cross + daily trend + RSI confirmation
            if fisher_bullish_cross and daily_bullish and rsi_bullish:
                new_signal = SIZE_ENTRY
            
            # Short: Fisher bearish cross + daily trend + RSI confirmation
            elif fisher_bearish_cross and daily_bearish and rsi_bearish:
                new_signal = -SIZE_ENTRY
        
        # === RANGING REGIME (CHOP > 61.8) ===
        # Mean reversion: fade Fisher extremes
        elif ranging_regime:
            # Long: Fisher oversold + RSI oversold
            if fisher_oversold and rsi[i] < 40:
                new_signal = SIZE_ENTRY
            
            # Short: Fisher overbought + RSI overbought
            elif fisher_overbought and rsi[i] > 60:
                new_signal = -SIZE_ENTRY
        
        # === NEUTRAL REGIME ===
        # Use Fisher crosses with stricter RSI filter
        else:
            if fisher_bullish_cross and rsi_neutral:
                new_signal = SIZE_ENTRY
            elif fisher_bearish_cross and rsi_neutral:
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