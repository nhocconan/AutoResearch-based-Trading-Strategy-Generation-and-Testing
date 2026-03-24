#!/usr/bin/env python3
"""
Experiment #082: 12h Primary + 1d HTF — Vol Regime + RSI Mean Reversion + Trend Filter

Hypothesis: 12h timeframe with volatility regime detection will capture both mean reversion
in choppy markets and trend following in directional markets. Using 1d HMA for HTF bias
ensures we trade with the higher timeframe trend. RSI extremes provide entry timing.

Key design choices:
1. 12h primary = fewer trades (target 20-50/year), less fee drag
2. Vol regime via ATR ratio (ATR7/ATR30) - high vol = mean revert, low vol = trend
3. 1d HMA for HTF trend bias - only trade in direction of daily trend
4. RSI(14) extremes for entry timing - oversold in uptrend, overbought in downtrend
5. SMA200 as additional trend filter
6. Loose enough thresholds to ensure 30+ trades/symbol (learned from 0-trade failures)

Target: Sharpe>0.351, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_volregime_rsi_trend_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_series = pd.Series(close)
    wma_half = close_series.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean().values
    wma_full = close_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    wma_diff = 2.0 * wma_half - wma_full
    hma = pd.Series(wma_diff).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    
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

def calculate_sma(close, period=200):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for HTF trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    sma_200 = calculate_sma(close, period=200)
    bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (discrete)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(sma_200[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY REGIME (ATR ratio) ===
        # High vol (ratio > 1.5) = mean reversion likely
        # Low vol (ratio < 1.2) = trending likely
        vol_ratio = atr_7[i] / atr_30[i]
        high_vol = vol_ratio > 1.3
        low_vol = vol_ratio < 1.1
        
        # === 12h TREND (HMA slope) ===
        hma_slope_bull = hma_12h[i] > hma_12h[i - 5] if not np.isnan(hma_12h[i - 5]) else False
        hma_slope_bear = hma_12h[i] < hma_12h[i - 5] if not np.isnan(hma_12h[i - 5]) else False
        
        # === SMA200 TREND FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi[i] < 35.0  # Looser threshold for more trades
        rsi_overbought = rsi[i] > 65.0  # Looser threshold for more trades
        
        # === BOLLINGER BAND POSITION ===
        near_bb_lower = close[i] < bb_lower[i] * 1.005 if not np.isnan(bb_lower[i]) else False
        near_bb_upper = close[i] > bb_upper[i] * 0.995 if not np.isnan(bb_upper[i]) else False
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # TREND FOLLOWING: Low vol + HTF + 12h trend alignment
        if low_vol and htf_bull and hma_slope_bull and above_sma200 and rsi[i] > 45:
            desired_signal = SIZE
        elif low_vol and htf_bear and hma_slope_bear and below_sma200 and rsi[i] < 55:
            desired_signal = -SIZE
        
        # MEAN REVERSION: High vol + RSI extremes + HTF bias
        # More aggressive entries to ensure trade count
        if high_vol:
            if htf_bull and rsi_oversold:
                desired_signal = SIZE
            elif htf_bear and rsi_overbought:
                desired_signal = -SIZE
        
        # BB mean reversion (additional signal path)
        if near_bb_lower and htf_bull:
            desired_signal = SIZE
        elif near_bb_upper and htf_bear:
            desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
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