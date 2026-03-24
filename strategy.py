#!/usr/bin/env python3
"""
Experiment #643: 6h Primary + 1d/1w HTF — Regime-Adaptive Dual Mode Strategy

Hypothesis: 6h timeframe sits between 4h and 12h - captures multi-day swings better than 4h
but with less lag than 12h. Key innovation: DUAL-MODE based on regime detection.

REGIME DETECTION:
- CHOP(14) > 61.8 + ADX(14) < 25 = RANGE mode → mean reversion at Bollinger extremes
- CHOP(14) < 38.2 + ADX(14) > 25 = TREND mode → Donchian breakout follow-through
- Otherwise = TRANSITION → reduce size 50%

ENTRY LOGIC:
- TREND mode: Donchian(20) breakout + ADX confirming + HTF bias alignment
- RANGE mode: RSI(14) < 30 or > 70 + price at Bollinger(20,2.5) extreme + HTF bias

HTF BIAS (1d + 1w HMA):
- Both bullish: full size long entries, half size short
- Both bearish: full size short entries, half size long
- Mixed: half size both directions

POSITION SIZING:
- Base: 0.25, Strong confluence: 0.30, Weak/transition: 0.15
- Discrete levels only: 0.0, ±0.15, ±0.25, ±0.30
- Stoploss: 2.5x ATR trailing

Target: Sharpe>0.40, trades>=30 train, trades>=3 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_regime_dual_mode_hma_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_chop(high, low, close, period=14):
    """Choppiness Index - identifies ranging vs trending markets"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        sum_tr = np.nansum(tr[i-period+1:i+1])
        
        if sum_tr > 1e-10 and (highest_high - lowest_low) > 1e-10:
            chop[i] = 100.0 * np.log10((highest_high - lowest_low) / sum_tr) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0.0)
        else:
            plus_dm[i] = 0.0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0.0)
        else:
            minus_dm[i] = 0.0
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_series = pd.Series(np.concatenate([[0], gain]))
    loss_series = pd.Series(np.concatenate([[0], loss]))
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_bollinger(close, period=20, std_mult=2.5):
    """Bollinger Bands with configurable std multiplier"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period):
    """Hull Moving Average for HTF"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    chop = calculate_chop(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    bb_upper, bb_lower = calculate_bollinger(close, period=20, std_mult=2.5)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        is_trend_regime = (chop[i] < 38.2) and (adx[i] > 25.0)
        is_range_regime = (chop[i] > 61.8) and (adx[i] < 25.0)
        is_transition = not is_trend_regime and not is_range_regime
        
        # === HTF BIAS (1d + 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # HTF confluence
        htf_strong_bull = htf_1d_bull and htf_1w_bull
        htf_strong_bear = htf_1d_bear and htf_1w_bear
        htf_mixed = (htf_1d_bull and htf_1w_bear) or (htf_1d_bear and htf_1w_bull)
        
        # === SIZE MULTIPLIER BASED ON HTF ===
        size_mult_long = 1.0 if htf_strong_bull else (0.5 if htf_mixed else 0.25)
        size_mult_short = 1.0 if htf_strong_bear else (0.5 if htf_mixed else 0.25)
        
        # === ENTRY SIGNALS ===
        desired_signal = 0.0
        signal_strength = 0  # 0=none, 1=weak, 2=strong
        
        if is_trend_regime:
            # TREND MODE: Donchian breakout
            breakout_long = close[i] >= donchian_upper[i] * 0.998  # slight buffer
            breakout_short = close[i] <= donchian_lower[i] * 1.002
            
            if breakout_long and adx[i] > 25.0:
                desired_signal = 0.30 * size_mult_long
                signal_strength = 2
            elif breakout_short and adx[i] > 25.0:
                desired_signal = -0.30 * size_mult_short
                signal_strength = 2
            # Pullback entry in trend
            elif htf_strong_bull and close[i] > hma_1d_aligned[i] and rsi[i] < 45:
                desired_signal = 0.25 * size_mult_long
                signal_strength = 1
            elif htf_strong_bear and close[i] < hma_1d_aligned[i] and rsi[i] > 55:
                desired_signal = -0.25 * size_mult_short
                signal_strength = 1
                
        elif is_range_regime:
            # RANGE MODE: Mean reversion at Bollinger extremes
            at_bb_lower = close[i] <= bb_lower[i] * 1.002
            at_bb_upper = close[i] >= bb_upper[i] * 0.998
            
            if at_bb_lower and rsi[i] < 35:
                desired_signal = 0.25 * size_mult_long
                signal_strength = 2
            elif at_bb_upper and rsi[i] > 65:
                desired_signal = -0.25 * size_mult_short
                signal_strength = 2
            # Weaker mean reversion
            elif rsi[i] < 30:
                desired_signal = 0.15 * size_mult_long
                signal_strength = 1
            elif rsi[i] > 70:
                desired_signal = -0.15 * size_mult_short
                signal_strength = 1
        
        else:
            # TRANSITION: Only take strong HTF-aligned signals, reduced size
            if htf_strong_bull and rsi[i] < 40 and close[i] > hma_1d_aligned[i]:
                desired_signal = 0.15
                signal_strength = 1
            elif htf_strong_bear and rsi[i] > 60 and close[i] < hma_1d_aligned[i]:
                desired_signal = -0.15
                signal_strength = 1
        
        # Reduce size in transition regime
        if is_transition and signal_strength > 0:
            desired_signal = desired_signal * 0.5
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
            signal_strength = 0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if signal_strength == 0 or abs(desired_signal) < 0.10:
            final_signal = 0.0
        elif desired_signal >= 0.25:
            final_signal = 0.30
        elif desired_signal <= -0.25:
            final_signal = -0.30
        elif desired_signal >= 0.15:
            final_signal = 0.25
        elif desired_signal <= -0.15:
            final_signal = -0.25
        elif desired_signal > 0:
            final_signal = 0.15
        else:
            final_signal = -0.15
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals