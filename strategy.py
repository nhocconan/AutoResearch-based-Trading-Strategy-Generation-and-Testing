#!/usr/bin/env python3
"""
Experiment #004: 4h Multi-Timeframe Supertrend Strategy with 1d HMA Bias
Hypothesis: 4h timeframe balances signal frequency with noise reduction. 
Uses 1d HMA for primary trend bias (slow, reliable), 4h Supertrend for entries,
4h RSI for pullback confirmation, and ADX to filter weak trends.
Multiple entry paths ensure >=10 trades per symbol across BTC/ETH/SOL.
Conservative sizing (0.25-0.30) with 2.5*ATR stoploss. Designed to work
through 2022 crash and 2025 bear market on ALL symbols.
Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_1d_hma_rsi_adx_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        if i == period:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            # If previous supertrend was upper band
            if direction[i-1] == 1:
                if close[i] > upper_band[i]:
                    supertrend[i] = upper_band[i]
                    direction[i] = 1
                else:
                    supertrend[i] = lower_band[i]
                    direction[i] = -1
            # If previous supertrend was lower band
            else:
                if close[i] < lower_band[i]:
                    supertrend[i] = lower_band[i]
                    direction[i] = -1
                else:
                    supertrend[i] = upper_band[i]
                    direction[i] = 1
    
    return supertrend, direction

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

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
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = np.inf
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - primary filter
        hma_1d_bullish = close[i] > hma_1d_aligned[i]
        hma_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # RSI zones - relaxed for more trades
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_neutral_long = rsi[i] > 35 and rsi[i] < 55
        rsi_neutral_short = rsi[i] > 45 and rsi[i] < 65
        rsi_turning_up = i > 0 and rsi[i] > rsi[i-1]
        rsi_turning_down = i > 0 and rsi[i] < rsi[i-1]
        
        # ADX trend strength - lower threshold for more trades
        trend_weak = adx[i] > 15
        trend_moderate = adx[i] > 20
        
        # EMA alignment
        ema_bullish = close[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: 1d bullish + Supertrend bullish + ADX moderate (trend follow)
        if hma_1d_bullish and st_bullish and trend_moderate:
            new_signal = SIZE_ENTRY
        
        # Path 2: 1d bullish + Supertrend bullish + RSI pullback (buy dip)
        elif hma_1d_bullish and st_bullish and rsi_neutral_long and rsi_turning_up:
            new_signal = SIZE_ENTRY
        
        # Path 3: 1d bullish + EMA bullish + RSI oversold bounce
        elif hma_1d_bullish and ema_bullish and rsi_oversold and rsi_turning_up:
            new_signal = SIZE_ENTRY
        
        # Path 4: Supertrend flip bullish + ADX building (new trend)
        elif st_bullish and st_direction[i-1] == -1 and adx[i] > 15:
            new_signal = SIZE_ENTRY
        
        # Path 5: 1d bullish + RSI oversold (dip buy in uptrend)
        elif hma_1d_bullish and rsi[i] < 35:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: 1d bearish + Supertrend bearish + ADX moderate (trend follow)
        if hma_1d_bearish and st_bearish and trend_moderate:
            new_signal = -SIZE_ENTRY
        
        # Path 2: 1d bearish + Supertrend bearish + RSI pullback (sell rip)
        elif hma_1d_bearish and st_bearish and rsi_neutral_short and rsi_turning_down:
            new_signal = -SIZE_ENTRY
        
        # Path 3: 1d bearish + EMA bearish + RSI overbought drop
        elif hma_1d_bearish and ema_bearish and rsi_overbought and rsi_turning_down:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Supertrend flip bearish + ADX building (new trend)
        elif st_bearish and st_direction[i-1] == 1 and adx[i] > 15:
            new_signal = -SIZE_ENTRY
        
        # Path 5: 1d bearish + RSI overbought (rip sell in downtrend)
        elif hma_1d_bearish and rsi[i] > 65:
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
                new_signal = SIZE_EXIT
            elif not position_reduced:
                # Take profit at 2R - reduce to half
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if current_stop < trailing_stop or trailing_stop == 0.0:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = SIZE_EXIT
            elif not position_reduced:
                # Take profit at 2R - reduce to half
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
            lowest_close = close[i] if position_side < 0 else np.inf
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else np.inf
            position_reduced = False
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = np.inf
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals