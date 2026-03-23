#!/usr/bin/env python3
"""
Experiment #481: 4h Primary + 1d/1w HTF — HMA Trend + Volume Breakout + RSI Entry

Hypothesis: Based on research showing Hull Moving Average (HMA) provides superior 
trend following with minimal lag compared to EMA/KAMA. Combined with volume 
confirmation for breakout validity and RSI for entry timing. Key innovations:
1. HMA(21/55) crossover - faster trend detection than KAMA, less whipsaw than EMA
2. Volume spike filter (vol > 1.5x 20-bar avg) - confirms breakout validity
3. RSI(14) entry timing - oversold/overbought with HTF bias alignment
4. 1d HMA for HTF major trend bias (cleaner than KAMA for trend direction)
5. 1w HMA for ultra-HTF regime filter (bull/bear market context)
6. ATR(14) trailing stop at 2.5x for risk management
7. Asymmetric sizing: 0.30 for trend-aligned, 0.20 for counter-trend mean revert
8. Relaxed entry thresholds to ensure 30-60 trades/year on 4h timeframe

Why this should work: HMA has proven superior in crypto trending markets (less lag).
Volume confirmation filters false breakouts (major issue in 4h timeframe). Dual HTF
bias (1d + 1w) provides strong trend context without over-complication. This is 
DIFFERENT from failed Fisher/Donchian combinations - using MA crossover + volume.
4h TF naturally targets 30-60 trades/year with proper entry filters.

Target: Sharpe > 0.612, DD < -35%, trades >= 30 on train, >= 3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_vol_rsi_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag significantly compared to EMA/SMA while maintaining smoothness.
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if period < 2:
        return hma
    
    # Helper function for WMA
    def wma(series, span):
        span = int(span)
        if span < 1:
            return np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # Calculate WMA components
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    if np.any(np.isnan(wma_half)) or np.any(np.isnan(wma_full)):
        return hma
    
    # 2*WMA(n/2) - WMA(n)
    raw_hma = 2.0 * wma_half - wma_full
    
    # Final WMA with sqrt period
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """
    Detect volume spikes relative to recent average.
    Returns ratio of current volume to rolling average.
    """
    n = len(volume)
    vol_ratio = np.full(n, np.nan)
    
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=period, min_periods=period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_ratio = volume / (vol_avg + 1e-10)
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 4h indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    hma_55 = calculate_hma(close, 55)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30  # Full size for trend-aligned trades
    SIZE_REVERT = 0.20  # Reduced size for counter-trend mean reversion
    SIZE_EXIT = 0.10  # Partial exit
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_55[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(vol_ratio[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === HTF MAJOR TREND BIAS (1w HMA) ===
        htf_bullish = close[i] > hma_1w_aligned[i]
        htf_bearish = close[i] < hma_1w_aligned[i]
        
        # === HTF INTERMEDIATE TREND (1d HMA) ===
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA crossover) ===
        hma_bullish = hma_21[i] > hma_55[i]
        hma_bearish = hma_21[i] < hma_55[i]
        
        # HMA slope confirmation
        hma_21_slope_up = hma_21[i] > hma_21[i - 3] if i >= 3 else False
        hma_21_slope_down = hma_21[i] < hma_21[i - 3] if i >= 3 else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5  # Volume 50% above average
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_extreme_oversold = rsi_14[i] < 30.0
        rsi_extreme_overbought = rsi_14[i] > 70.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        long_score = 0
        
        # HTF bias alignment (1w + 1d both bullish = strong signal)
        if htf_bullish and htf_1d_bullish:
            long_score += 3
        elif htf_bullish or htf_1d_bullish:
            long_score += 1
        
        # Primary trend alignment
        if hma_bullish:
            long_score += 2
        
        # HMA slope confirmation
        if hma_21_slope_up:
            long_score += 1
        
        # RSI entry signal
        if rsi_oversold:
            long_score += 1
        if rsi_extreme_oversold:
            long_score += 2
        
        # Volume confirmation (adds confidence but not required)
        if vol_spike and hma_bullish:
            long_score += 1
        
        # Enter long if score >= 5 (relaxed for trade generation)
        if long_score >= 5:
            # Use full size if HTF aligned, reduced if counter-trend
            if htf_bullish and htf_1d_bullish:
                desired_signal = SIZE_TREND
            else:
                desired_signal = SIZE_REVERT
        
        # SHORT ENTRIES
        if desired_signal == 0.0:
            short_score = 0
            
            # HTF bias alignment
            if htf_bearish and htf_1d_bearish:
                short_score += 3
            elif htf_bearish or htf_1d_bearish:
                short_score += 1
            
            # Primary trend alignment
            if hma_bearish:
                short_score += 2
            
            # HMA slope confirmation
            if hma_21_slope_down:
                short_score += 1
            
            # RSI entry signal
            if rsi_overbought:
                short_score += 1
            if rsi_extreme_overbought:
                short_score += 2
            
            # Volume confirmation
            if vol_spike and hma_bearish:
                short_score += 1
            
            if short_score >= 5:
                if htf_bearish and htf_1d_bearish:
                    desired_signal = -SIZE_TREND
                else:
                    desired_signal = -SIZE_REVERT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and hma_bullish and htf_bullish:
                desired_signal = SIZE_TREND if htf_1d_bullish else SIZE_REVERT
            elif position_side < 0 and hma_bearish and htf_bearish:
                desired_signal = -SIZE_TREND if htf_1d_bearish else -SIZE_REVERT
        
        # === EXIT LOGIC — RSI extreme reversal ===
        if in_position and desired_signal != 0.0:
            if position_side > 0 and rsi_14[i] > 75.0:
                # Partial exit on overbought
                desired_signal = SIZE_EXIT
            elif position_side < 0 and rsi_14[i] < 25.0:
                # Partial exit on oversold
                desired_signal = -SIZE_EXIT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.15:
            desired_signal = 0.30
        elif desired_signal > 0.0:
            desired_signal = 0.10
        elif desired_signal < -0.15:
            desired_signal = -0.30
        elif desired_signal < 0.0:
            desired_signal = -0.10
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals