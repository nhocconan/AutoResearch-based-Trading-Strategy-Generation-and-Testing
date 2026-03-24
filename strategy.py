#!/usr/bin/env python3
"""
Experiment #162: 4h Primary + 1d/1w HTF — HMA Trend + Choppiness Regime + RSI Entry

Hypothesis: 4h timeframe with daily/weekly HTF bias provides optimal trade frequency
(20-50 trades/year) while avoiding fee drag from lower timeframes. Choppiness Index
filters entries to avoid whipsaw in ranging markets. RSI provides entry timing.

Key design decisions:
- 4h HMA(21) for primary trend direction
- 1d HMA(50) for daily bias confirmation
- 1w HMA(50) for weekly regime filter
- Choppiness Index(14) to detect ranging vs trending markets
- RSI(14) with moderate thresholds (40/60) for entry timing
- ATR(14) 2.5x trailing stop for risk management
- Position size: 0.28 (28% of capital)

Trade generation strategy (CRITICAL - avoid 0 trades):
- Multiple entry paths with decreasing size requirements
- Primary: All HTF aligned + CHOP filter + RSI confirmation
- Fallback 1: Strong 4h+1d alignment (ignore 1w)
- Fallback 2: Very strong 4h momentum alone
- Target 25-45 trades/year on 4h timeframe

Target: Sharpe>0.167, DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_chop_rsi_1d1w_v1"
timeframe = "4h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR for each bar
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for daily bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for weekly regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after 200 SMA is ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
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
        if np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === WEEKLY REGIME (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP < 45 = trending (favor trend entries)
        # CHOP > 55 = ranging (favor mean reversion, skip trend entries)
        trending_market = chop[i] < 45.0
        ranging_market = chop[i] > 55.0
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI CONFIRMATION (moderate thresholds) ===
        rsi_ok_long = rsi[i] > 40.0  # not oversold
        rsi_ok_short = rsi[i] < 60.0  # not overbought
        rsi_strong_long = rsi[i] > 50.0
        rsi_strong_short = rsi[i] < 50.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # PRIMARY: All HTF aligned + trending market + RSI confirmation
        # Long: 4h HMA bull + 1d HMA bull + 1w HMA bull + trending + RSI ok + above SMA200
        if hma_bull and htf_1d_bull and htf_1w_bull and trending_market and rsi_ok_long and above_sma200:
            desired_signal = SIZE
        
        # Short: 4h HMA bear + 1d HMA bear + 1w HMA bear + trending + RSI ok + below SMA200
        elif hma_bear and htf_1d_bear and htf_1w_bear and trending_market and rsi_ok_short and below_sma200:
            desired_signal = -SIZE
        
        # FALLBACK 1: Strong 4h + 1d alignment (ignore 1w, ignore CHOP) - 80% size
        elif hma_bull and htf_1d_bull and rsi_strong_long and above_sma200:
            desired_signal = SIZE * 0.8
        
        elif hma_bear and htf_1d_bear and rsi_strong_short and below_sma200:
            desired_signal = -SIZE * 0.8
        
        # FALLBACK 2: 4h HMA + RSI strong (ignore HTF) - 60% size
        # Ensures trades in transitional markets
        elif hma_bull and rsi[i] > 55.0 and above_sma200:
            desired_signal = SIZE * 0.6
        
        elif hma_bear and rsi[i] < 45.0 and below_sma200:
            desired_signal = -SIZE * 0.6
        
        # FALLBACK 3: Very strong momentum breakout - 40% size
        # Catches strong moves even when indicators conflicted
        elif hma_bull and rsi[i] > 60.0:
            desired_signal = SIZE * 0.4
        
        elif hma_bear and rsi[i] < 40.0:
            desired_signal = -SIZE * 0.4
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.7:
            final_signal = SIZE * 0.8
        elif desired_signal <= -SIZE * 0.7:
            final_signal = -SIZE * 0.8
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.6
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.6
        elif desired_signal >= SIZE * 0.3:
            final_signal = SIZE * 0.4
        elif desired_signal <= -SIZE * 0.3:
            final_signal = -SIZE * 0.4
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