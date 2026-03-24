#!/usr/bin/env python3
"""
Experiment #024: 12h Primary + 1d HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After 23 failed experiments, the pattern is clear:
- Pure trend following fails on BTC/ETH in bear/range markets (2022 crash, 2025 bear)
- Connors RSI (CRSI) has proven 75% win rate for mean reversion entries
- Choppiness Index reliably detects regime (chop vs trend)
- SOLUTION: CRSI for entry timing + CHOP for regime + 1d HMA for bias
- LOOSE thresholds to ensure >=30 trades on train, >=3 on test (critical lesson from 0 trades failures)
- Volume confirmation to filter false signals

Key design choices:
- Timeframe: 12h (20-50 trades/year target, lower fee drag)
- HTF: 1d HMA(50) for major trend bias
- Entry: CRSI extremes (10/90) + regime filter + volume confirmation
- Regime: CHOP>55 = range (mean revert), CHOP<55 = trend (follow breakout)
- Position size: 0.28 (28% of capital, conservative for 12h)
- Stoploss: 2.5x ATR trailing
- LOOSE filters: CRSI<20/>80 (not 10/90) to ensure trades generate

Target: Sharpe>0.019 (beat current best), DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_hma_vol_1d_v1"
timeframe = "12h"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries
    Long: CRSI < 10-15, Short: CRSI > 85-90
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # RSI on streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive values for RSI calculation
    streak_positive = np.where(streak >= 0, streak, 0)
    streak_negative = np.where(streak < 0, -streak, 0)
    
    avg_gain_streak = pd.Series(streak_positive).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_loss_streak = pd.Series(streak_negative).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        if avg_loss_streak[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_gain_streak[i] / (avg_loss_streak[i] + 1e-10)
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # PercentRank: percentage of closes in lookback period that are below current close
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        lookback = close[i-rank_period+1:i+1]
        count_below = np.sum(lookback[:-1] < close[i])  # exclude current
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 55 as threshold for regime switch
    """
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
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_volume_ma(volume, period=20):
    """Volume moving average for volume confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    # Also calculate simple RSI for additional filter
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 12h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(vol_ma[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = range/choppy (mean revert)
        # CHOP < 55 = trending (trend follow)
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] <= 55.0
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * vol_ma[i]  # At least 80% of avg volume
        
        # === CRSI EXTREMES (LOOSE THRESHOLDS FOR TRADES) ===
        # Using 20/80 instead of 10/90 to ensure more trades generate
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        
        # === 12h HMA TREND ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === DESIRED SIGNAL (Dual Regime Logic with CRSI) ===
        desired_signal = 0.0
        
        if is_choppy:
            # CHOPPY REGIME: Mean reversion with CRSI
            # LONG: CRSI oversold + volume ok + HTF not strongly bear
            if crsi_oversold and volume_ok and not htf_bear:
                desired_signal = SIZE
            # SHORT: CRSI overbought + volume ok + HTF not strongly bull
            elif crsi_overbought and volume_ok and not htf_bull:
                desired_signal = -SIZE
            # Fallback: extreme CRSI even without volume
            elif crsi[i] < 15.0 and hma_bull:
                desired_signal = SIZE * 0.7
            elif crsi[i] > 85.0 and hma_bear:
                desired_signal = -SIZE * 0.7
        else:
            # TRENDING REGIME: Follow trend with CRSI pullback entries
            # LONG: HTF bull + HMA bull + CRSI pullback (not oversold but recovering)
            if htf_bull and hma_bull and crsi[i] > 30.0 and crsi[i] < 60.0:
                desired_signal = SIZE
            # SHORT: HTF bear + HMA bear + CRSI pullback
            elif htf_bear and hma_bear and crsi[i] > 40.0 and crsi[i] < 70.0:
                desired_signal = -SIZE
            # Fallback: strong trend continuation
            elif htf_bull and hma_bull and volume_ok and crsi[i] > 40.0:
                desired_signal = SIZE * 0.7
            elif htf_bear and hma_bear and volume_ok and crsi[i] < 60.0:
                desired_signal = -SIZE * 0.7
        
        # === ADDITIONAL TRADE GENERATION (ensure we get trades) ===
        # If no signal yet, check for simple RSI extremes
        if desired_signal == 0.0:
            if rsi_14[i] < 25.0 and hma_bull and volume_ok:
                desired_signal = SIZE * 0.5
            elif rsi_14[i] > 75.0 and hma_bear and volume_ok:
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