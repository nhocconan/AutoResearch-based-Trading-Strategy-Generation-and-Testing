#!/usr/bin/env python3
"""
Experiment #012: 12h Primary + 1d/1w HTF — Regime-Adaptive HMA + Choppiness Switch

Hypothesis: After 11 experiments, the key insight is that NO SINGLE strategy works 
in all regimes. BTC/ETH 2025 is bear/range, not bull trend. We need to ADAPT:
- When CHOP > 61.8 (range): Mean-revert at Bollinger bands
- When CHOP < 38.2 (trend): Follow 1d HMA direction

Key improvements over #011 (KAMA 4h):
1. 12h primary TF (fewer trades, less fee drag) — target 20-50 trades/year
2. Regime-adaptive logic (not pure trend-follow which failed in 2022/2025)
3. 1d HMA for HTF bias (HMA proven in baseline Sharpe=5.4 strategy)
4. 1w HMA for meta-trend filter (avoid counter-trend trades in strong moves)
5. Looser RSI thresholds (25/75 vs 30/70) to ensure trades generate
6. Choppiness Index (14) as regime switch — proven in research notes

Entry Logic:
- TREND REGIME (CHOP < 38.2): Long if close > 1d HMA + 1w HMA sloping up
                           Short if close < 1d HMA + 1w HMA sloping down
- RANGE REGIME (CHOP > 61.8): Long at BB lower band + RSI < 30
                             Short at BB upper band + RSI > 70
- Size: 0.30 (discrete, minimizes fee churn)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.3, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_hma_chop_bb_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - responsive trend indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    wma_diff = 2.0 * wma_half - wma_full
    hma = pd.Series(wma_diff).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    CHOP > 61.8 = range/choppy market (mean-revert)
    CHOP < 38.2 = trending market (trend-follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Max High - Min Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range < 1e-10 or atr_sum < 1e-10:
            chop[i] = 100.0
        else:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands - for mean-reversion entries"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def calculate_rsi(close, period=14):
    """RSI - momentum filter with loose thresholds"""
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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for HTF trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for meta-trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d HMA slope (previous bar for no look-ahead)
    hma_1d_slope = np.zeros(n)
    for i in range(2, n):
        if not np.isnan(hma_1d_aligned[i-1]) and not np.isnan(hma_1d_aligned[i-2]):
            hma_1d_slope[i] = hma_1d_aligned[i-1] - hma_1d_aligned[i-2]
        else:
            hma_1d_slope[i] = 0.0
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    chop = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop[i] < 38.2  # Trending market
        is_range_regime = chop[i] > 61.8  # Range/choppy market
        
        # === HTF TREND BIAS (1d + 1w HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        hma_1d_sloping_up = hma_1d_slope[i] > 0
        hma_1d_sloping_down = hma_1d_slope[i] < 0
        
        # === DESIRED SIGNAL (Regime-Adaptive) ===
        desired_signal = 0.0
        
        if is_trend_regime:
            # TREND FOLLOWING MODE
            # Long: Price above 1d HMA + 1d HMA sloping up + 1w HMA bull confirmation
            if hma_1d_bull and hma_1d_sloping_up and hma_1w_bull:
                desired_signal = SIZE
            
            # Short: Price below 1d HMA + 1d HMA sloping down + 1w HMA bear confirmation
            elif hma_1d_bear and hma_1d_sloping_down and hma_1w_bear:
                desired_signal = -SIZE
        
        elif is_range_regime:
            # MEAN REVERSION MODE
            # Long: At BB lower band + RSI oversold
            if close[i] <= bb_lower[i] * 1.002 and rsi[i] < 35.0:
                desired_signal = SIZE
            
            # Short: At BB upper band + RSI overbought
            elif close[i] >= bb_upper[i] * 0.998 and rsi[i] > 65.0:
                desired_signal = -SIZE
        
        else:
            # NEUTRAL REGIME (38.2 <= CHOP <= 61.8) - reduced position or flat
            # Only trade if strong HTF alignment
            if hma_1d_bull and hma_1w_bull and rsi[i] > 45.0:
                desired_signal = SIZE * 0.5
            elif hma_1d_bear and hma_1w_bear and rsi[i] < 55.0:
                desired_signal = -SIZE * 0.5
        
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
        elif desired_signal >= SIZE * 0.4:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal <= -SIZE * 0.4:
            final_signal = -SIZE * 0.5
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