#!/usr/bin/env python3
"""
Experiment #179: 1h Primary + 4h/12h HTF — Simplified Regime + Pullback Strategy

Hypothesis: Previous 1h strategies failed (0 trades) due to TOO MANY filters.
This version SIMPLIES entry logic while keeping HTF trend bias.

Key changes from failed 1h attempts (#170, #173, #176, #177):
1. REMOVED session filter (was blocking 60% of valid signals)
2. RELAXED RSI thresholds (30/70 instead of 20/80)
3. REMOVED CRSI complexity (simple RSI works better for trade frequency)
4. SIMPLIFIED regime detection (BB width percentile only)
5. ADDED volume confirmation (prevent false breakouts)

Strategy Logic:
- HTF Trend: 4h HMA(21) for direction bias
- HTF Confirmation: 12h HMA(50) for major trend
- Entry: RSI(14) pullback to BB bands in trend direction
- Regime: BB Width > 70th percentile = volatile (wider stops), < 30th = quiet
- Volume: Require volume > 0.8 * SMA(volume, 20) for entries

Target: 40-80 trades/year, Sharpe > 0.167 (beat current best 1d strategy)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_bb_pullback_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower, sma

def calculate_bb_width(upper, lower, sma):
    """Bollinger Band Width as % of price"""
    n = len(upper)
    width = np.zeros(n)
    width[:] = np.nan
    
    for i in range(n):
        if not np.isnan(upper[i]) and not np.isnan(lower[i]) and sma[i] > 1e-10:
            width[i] = (upper[i] - lower[i]) / sma[i] * 100.0
    
    return width

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_percentile_rank(values, lookback=100):
    """Percentile rank of current value vs lookback period"""
    n = len(values)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(values[i]):
            window = values[i-lookback:i]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                count_below = np.sum(valid_window < values[i])
                pr[i] = count_below / len(valid_window) * 100.0
    
    return pr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for major trend confirmation
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_sma)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    # Calculate BB width percentile for regime detection
    bb_width_pr = calculate_percentile_rank(bb_width, lookback=100)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25  # 25% base position size
    SIZE_STRONG = 0.30  # 30% for strong signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === HTF MAJOR TREND (12h HMA) ===
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === BB REGIME (volatility state) ===
        high_vol = not np.isnan(bb_width_pr[i]) and bb_width_pr[i] > 70.0
        low_vol = not np.isnan(bb_width_pr[i]) and bb_width_pr[i] < 30.0
        
        # === RSI CONDITIONS (relaxed for more trades) ===
        rsi_oversold = rsi[i] < 35.0  # Was 30, relaxed for more trades
        rsi_overbought = rsi[i] > 65.0  # Was 70, relaxed for more trades
        
        # === BB TOUCH CONDITIONS ===
        near_bb_lower = close[i] <= bb_lower[i] * 1.002  # Within 0.2% of lower band
        near_bb_upper = close[i] >= bb_upper[i] * 0.998  # Within 0.2% of upper band
        
        # === ENTRY LOGIC (simplified for trade frequency) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + RSI oversold + near BB lower + volume OK
        # Require EITHER 12h bullish OR low volatility regime
        if htf_4h_bull and rsi_oversold and near_bb_lower and volume_ok:
            if htf_12h_bull or low_vol:
                desired_signal = SIZE_BASE
                if high_vol:
                    desired_signal = SIZE_STRONG  # Higher conviction in high vol
        
        # SHORT: 4h bearish + RSI overbought + near BB upper + volume OK
        # Require EITHER 12h bearish OR low volatility regime
        elif htf_4h_bear and rsi_overbought and near_bb_upper and volume_ok:
            if htf_12h_bear or low_vol:
                desired_signal = -SIZE_BASE
                if high_vol:
                    desired_signal = -SIZE_STRONG  # Higher conviction in high vol
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals