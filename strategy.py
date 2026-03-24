#!/usr/bin/env python3
"""
Experiment #483: 6h Primary + 1d/1w HTF — Simplified Trend + Pullback Hybrid

Hypothesis: 6h is unexplored territory between 4h (proven) and 12h (often fails).
Key insight from #480 (Sharpe=0.086): Chop+Connors+HMA on 6h with 1d/1w almost worked.
This strategy SIMPLIFIES that pattern:

1. 1w HMA(21) = ultra-long-term bias (only trade long when price > 1w HMA)
2. 1d HMA(21) = medium-term trend confirmation
3. 6h RSI(14) pullback to 40-50 zone in uptrend = long entry (proven from 4h)
4. 6h Donchian(20) breakout with volume confirmation = trend entry
5. 6h RSI(14) > 60 in downtrend = short entry
6. ATR(14)*2.0 stoploss on all positions
7. Discrete signals: 0.0, ±0.25, ±0.30

Key changes from failed 6h experiments:
- LOOSE RSI thresholds (40/60 not 30/70) to ensure trade generation
- Volume confirmation ONLY on breakouts (not on pullbacks)
- OR logic for entries (any trigger works)
- Dual HTF (1w + 1d) for stronger bias but not overly restrictive

Target: Sharpe>0.40, trades>=30 train, trades>=3 test
Timeframe: 6h (FIRST 6h experiment with proper MTF)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_rsi_donchian_1d1w_v1"
timeframe = "6h"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for ultra-long bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d HMA for medium-term trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]):
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
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_50[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1w HTF ULTRA-LONG BIAS ===
        htf_weekly_bull = close[i] > hma_1w_aligned[i]
        htf_weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d HTF MEDIUM-TERM TREND ===
        htf_daily_bull = close[i] > hma_1d_aligned[i]
        htf_daily_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === RSI ZONES (LOOSE: 40/60 for entries) ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        rsi_neutral_long = 40.0 <= rsi[i] <= 55.0
        rsi_neutral_short = 45.0 <= rsi[i] <= 65.0
        rsi_extreme_oversold = rsi[i] < 30.0
        rsi_extreme_overbought = rsi[i] > 70.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakdown_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === VOLUME CONFIRMATION ===
        volume_above_avg = volume[i] > vol_sma[i] * 1.2
        
        # === ENTRY LOGIC (LOOSE - OR logic, not AND) ===
        desired_signal = 0.0
        
        # TREND LONG: 1w bull + 1d bull + (RSI pullback OR Donchian breakout)
        if htf_weekly_bull and htf_daily_bull:
            if donchian_breakout_long and volume_above_avg:
                desired_signal = SIZE_STRONG
            elif rsi_neutral_long and hma_bull and above_sma50:
                # RSI pullback in uptrend
                desired_signal = SIZE_BASE
            elif rsi[i] > 45.0 and rsi[i-1] <= 45.0 and above_sma50:
                # RSI crossing above 45 = momentum shift
                desired_signal = SIZE_BASE
        
        # TREND SHORT: 1w bear + 1d bear + (RSI weakness OR Donchian breakdown)
        elif htf_weekly_bear and htf_daily_bear:
            if donchian_breakdown_short and volume_above_avg:
                desired_signal = -SIZE_STRONG
            elif rsi_neutral_short and hma_bear and below_sma50:
                # RSI pullback in downtrend
                desired_signal = -SIZE_BASE
            elif rsi[i] < 55.0 and rsi[i-1] >= 55.0 and below_sma50:
                # RSI crossing below 55 = weakness
                desired_signal = -SIZE_BASE
        
        # MEAN REVERSION LONG: RSI extreme (weaker HTF requirement)
        if desired_signal == 0.0:
            if rsi_extreme_oversold and above_sma200:
                desired_signal = SIZE_BASE
            elif rsi_oversold and htf_daily_bull:
                # Oversold in daily uptrend
                desired_signal = SIZE_BASE * 0.8
        
        # MEAN REVERSION SHORT: RSI extreme (weaker HTF requirement)
        if desired_signal == 0.0:
            if rsi_extreme_overbought and below_sma200:
                desired_signal = -SIZE_BASE
            elif rsi_overbought and htf_daily_bear:
                # Overbought in daily downtrend
                desired_signal = -SIZE_BASE * 0.8
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals