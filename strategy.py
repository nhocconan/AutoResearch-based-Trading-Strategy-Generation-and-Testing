#!/usr/bin/env python3
"""
Experiment #369: 15m Primary + 1h/1d HTF — Daily Pivot + HMA Trend + RSI Pullback

Hypothesis: Previous 15m strategies failed due to (a) 0 trades from overly strict filters,
or (b) too many trades causing fee drag. This version balances selectivity with trigger frequency.

Key design:
1. 1d Central Pivot Range (CPR) for concrete S/R levels - price action respects these
2. 1h HMA(21) for trend bias - only trade in HTF trend direction
3. 15m RSI(7) for entry timing - quick mean reversion within trend
4. Volume confirmation (1.3x SMA) to avoid fake breakouts
5. Session preference 00-14 UTC (London+NY overlap) but NOT mandatory

Entry Logic:
- Long: 1h HMA bull + price > daily pivot + RSI(7) < 35 then crosses above 35 + vol confirm
- Short: 1h HMA bear + price < daily pivot + RSI(7) > 65 then crosses below 65 + vol confirm

Position sizing: 0.15 base, 0.20 when 1d trend aligned (conservative for 15m frequency)
Stoploss: 2.5x ATR(14) from entry
Target: 50-80 trades/year, Sharpe > 0.40, DD > -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_daily_pivot_hma_rsi_1h1d_v1"
timeframe = "15m"
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

def calculate_daily_pivot(df_1d):
    """
    Calculate Daily Pivot levels from 1d data.
    Returns: pivot, r1, s1, r2, s2, tc, bc (all aligned arrays)
    Classic pivot: P = (H + L + C) / 3
    CPR: TC = (P + BC) / 2, BC = (H + L) / 2
    """
    n_1d = len(df_1d)
    
    high = df_1d['high'].values
    low = df_1d['low'].values
    close = df_1d['close'].values
    
    pivot = np.zeros(n_1d)
    r1 = np.zeros(n_1d)
    s1 = np.zeros(n_1d)
    r2 = np.zeros(n_1d)
    s2 = np.zeros(n_1d)
    tc = np.zeros(n_1d)  # Top Central Pivot
    bc = np.zeros(n_1d)  # Bottom Central Pivot
    
    for i in range(1, n_1d):
        h = high[i-1]  # Previous day's high
        l = low[i-1]   # Previous day's low
        c = close[i-1] # Previous day's close
        
        pivot[i] = (h + l + c) / 3.0
        bc[i] = (h + l) / 2.0
        tc[i] = (pivot[i] + bc[i]) / 2.0
        
        r1[i] = 2.0 * pivot[i] - l
        s1[i] = 2.0 * pivot[i] - h
        r2[i] = pivot[i] + (h - l)
        s2[i] = pivot[i] - (h - l)
    
    # First bar has no previous day data
    pivot[0] = np.nan
    r1[0] = np.nan
    s1[0] = np.nan
    r2[0] = np.nan
    s2[0] = np.nan
    tc[0] = np.nan
    bc[0] = np.nan
    
    return pivot, r1, s1, r2, s2, tc, bc

def calculate_volume_sma(volume, period=20):
    """Volume SMA for confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias (1h)
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate and align daily pivot levels (1d)
    pivot_1d, r1_1d, s1_1d, r2_1d, s2_1d, tc_1d, bc_1d = calculate_daily_pivot(df_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    tc_aligned = align_htf_to_ltf(prices, df_1d, tc_1d)
    bc_aligned = align_htf_to_ltf(prices, df_1d, bc_1d)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    # RSI cross tracking
    prev_rsi_7 = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (1h HMA) ===
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        
        # === DAILY PIVOT POSITION ===
        above_pivot = close[i] > pivot_aligned[i]
        below_pivot = close[i] < pivot_aligned[i]
        
        # CPR width (narrow CPR = potential breakout day)
        cpr_width = np.nan
        if not np.isnan(tc_aligned[i]) and not np.isnan(bc_aligned[i]):
            cpr_width = abs(tc_aligned[i] - bc_aligned[i]) / pivot_aligned[i] * 100.0
        
        narrow_cpr = cpr_width < 0.5 if not np.isnan(cpr_width) else False
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === RSI CROSSOVER (mean reversion entry) ===
        rsi_cross_up_35 = False
        rsi_cross_down_65 = False
        
        if i > 0 and not np.isnan(rsi_7[i]) and not np.isnan(prev_rsi_7):
            if prev_rsi_7 < 35.0 and rsi_7[i] >= 35.0:
                rsi_cross_up_35 = True
            if prev_rsi_7 > 65.0 and rsi_7[i] <= 65.0:
                rsi_cross_down_65 = True
        
        prev_rsi_7 = rsi_7[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_7[i] < 30.0
        rsi_overbought = rsi_7[i] > 70.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = False
        if not np.isnan(vol_sma[i]) and vol_sma[i] > 1e-10:
            vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        # === SESSION FILTER (prefer 00-14 UTC, but not mandatory) ===
        # 15m bars: 00:00 = bar 0, 00:15 = bar 1, etc.
        # 14:00 UTC = bar 56 (14 * 4)
        bar_of_day = i % 96  # 96 bars per day on 15m
        preferred_session = bar_of_day < 56  # 00:00 - 14:00 UTC
        
        # === ENTRY LOGIC (3+ confluence required) ===
        desired_signal = 0.0
        
        # LONG: 1h bull + above pivot + RSI cross up OR RSI oversold bounce
        long_confluence = 0
        if htf_1h_bull:
            long_confluence += 1
        if above_pivot:
            long_confluence += 1
        if hma_bull:
            long_confluence += 1
        if rsi_cross_up_35 or rsi_oversold:
            long_confluence += 1
        if vol_confirm:
            long_confluence += 1
        
        # Need at least 3 confluence for long
        if long_confluence >= 3 and (rsi_cross_up_35 or rsi_oversold):
            if htf_1h_bull and above_pivot:
                desired_signal = SIZE_STRONG if vol_confirm else SIZE_BASE
        
        # SHORT: 1h bear + below pivot + RSI cross down OR RSI overbought
        short_confluence = 0
        if htf_1h_bear:
            short_confluence += 1
        if below_pivot:
            short_confluence += 1
        if hma_bear:
            short_confluence += 1
        if rsi_cross_down_65 or rsi_overbought:
            short_confluence += 1
        if vol_confirm:
            short_confluence += 1
        
        # Need at least 3 confluence for short
        if short_confluence >= 3 and (rsi_cross_down_65 or rsi_overbought):
            if htf_1h_bear and below_pivot:
                desired_signal = -SIZE_STRONG if vol_confirm else -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
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
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals