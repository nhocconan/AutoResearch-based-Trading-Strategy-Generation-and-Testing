#!/usr/bin/env python3
"""
Experiment #005: 15m Primary + 4h/1d HTF — CPR Pivot + Volume + RSI Confluence

Hypothesis: 15m strategies fail due to excessive trades and fee drag. Solution:
- 4h HMA(21) for major trend bias (proven in successful multi-TF strategies)
- 1d CPR (Central Pivot Range) for institutional pivot levels - key S/R
- Volume confirmation: volume > 1.3x 20-bar average (filters fakeouts)
- RSI(7) for faster 15m entry timing (more responsive than RSI(14))
- 3+ confluence required: HTF trend + CPR level + volume + RSI
- Position size: 0.20 (conservative for 15m frequency)
- Target: 50-100 trades/year, Sharpe > 0.3, ALL symbols positive

Key innovations vs failed attempts:
1. Daily CPR from 1d HTF - institutions watch these levels (unlike simple EMA)
2. Volume spike confirmation ensures real breakouts not fakeouts
3. Fast RSI(7) for 15m responsiveness (RSI(14) too slow for 15m)
4. Strict confluence keeps trade count low (avoid 300+ trades/year)
5. Looser than #001 to ensure trades generate (learned from Sharpe=0.000 failure)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_cpr_volume_rsi_4h1d_v1"
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
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_cpr_1d(df_1d):
    """
    Central Pivot Range from daily data
    TC = (Pivot + BC) / 2
    Pivot = (High + Low + Close) / 3
    BC = (High + Low) / 2
    Returns arrays aligned to 1d bars
    """
    high = df_1d['high'].values
    low = df_1d['low'].values
    close = df_1d['close'].values
    n = len(close)
    
    pivot = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = (pivot + bc) / 2.0
    
    return pivot, bc, tc

def calculate_volume_sma(volume, period=20):
    """Volume simple moving average"""
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
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d CPR levels
    pivot_1d_raw, bc_1d_raw, tc_1d_raw = calculate_cpr_1d(df_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_raw)
    bc_1d_aligned = align_htf_to_ltf(prices, df_1d, bc_1d_raw)
    tc_1d_aligned = align_htf_to_ltf(prices, df_1d, tc_1d_raw)
    
    # Calculate primary (15m) indicators
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (conservative for 15m frequency)
    
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
        if np.isnan(rsi[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(pivot_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / (vol_sma[i] + 1e-10)
        volume_confirmed = vol_ratio > 1.3  # 30% above average
        
        # === 4h TREND BIAS ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 1d CPR LEVELS ===
        # Narrow CPR = potential breakout day (TC - BC < 0.8% of price)
        cpr_width = (tc_1d_aligned[i] - bc_1d_aligned[i]) / (pivot_1d_aligned[i] + 1e-10)
        narrow_cpr = cpr_width < 0.008
        
        # Price position relative to CPR
        above_cpr = close[i] > tc_1d_aligned[i]
        below_cpr = close[i] < bc_1d_aligned[i]
        inside_cpr = bc_1d_aligned[i] <= close[i] <= tc_1d_aligned[i]
        
        # Distance to CPR levels (for mean reversion entries)
        dist_to_tc = (tc_1d_aligned[i] - close[i]) / (close[i] + 1e-10)
        dist_to_bc = (close[i] - bc_1d_aligned[i]) / (close[i] + 1e-10)
        near_tc = dist_to_tc > -0.01 and dist_to_tc < 0.015  # Within 1-1.5% of TC
        near_bc = dist_to_bc > -0.01 and dist_to_bc < 0.015  # Within 1-1.5% of BC
        
        # === RSI SIGNALS (Fast RSI(7)) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_neutral = 40.0 <= rsi[i] <= 60.0
        
        # === DESIRED SIGNAL (Multiple Confluence) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + volume + (above CPR pullback OR near BC bounce OR narrow CPR breakout)
        if htf_bull and volume_confirmed:
            if above_cpr and rsi_oversold:
                # Pullback above CPR in uptrend + oversold RSI
                desired_signal = SIZE
            elif near_bc and rsi[i] < 45.0:
                # Bounce off BC support + RSI recovering
                desired_signal = SIZE
            elif narrow_cpr and above_cpr and rsi_neutral:
                # Narrow CPR breakout with bullish bias
                desired_signal = SIZE * 0.7
        
        # SHORT: 4h bear + volume + (below CPR rally OR near TC reject OR narrow CPR breakdown)
        if htf_bear and volume_confirmed:
            if below_cpr and rsi_overbought:
                # Rally below CPR in downtrend + overbought RSI
                desired_signal = -SIZE
            elif near_tc and rsi[i] > 55.0:
                # Reject at TC resistance + RSI peaking
                desired_signal = -SIZE
            elif narrow_cpr and below_cpr and rsi_neutral:
                # Narrow CPR breakdown with bearish bias
                desired_signal = -SIZE * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR 2x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
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