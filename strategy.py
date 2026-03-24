#!/usr/bin/env python3
"""
Experiment #708: 4h Primary + 12h/1d HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: 4h timeframe with regime-adaptive logic will outperform single-regime strategies.
Using Choppiness Index to detect range vs trend, then applying Connors RSI for mean reversion
in ranges and HMA/Donchian for trend following. 12h/1d HTF provides directional bias.

Key innovations:
1. Choppiness Index (14) - regime detection: >61.8=range, <38.2=trend
2. Connors RSI - (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 for mean reversion
3. HMA(21/63) crossover - trend direction confirmation
4. Donchian(20) breakout - trend entry trigger
5. 12h HMA(21) + 1d HMA(21) - HTF bias confirmation
6. ATR(14) 2.5x trailing stop - risk management
7. Discrete sizing: 0.0, ±0.20, ±0.30 to minimize fee churn

Entry logic (regime-adaptive):
- RANGE (CHOP>61.8): Connors RSI <10 long, >90 short (mean reversion)
- TREND (CHOP<38.2): HMA bull + Donchian breakout long, HMA bear + Donchian breakdown short
- HTF bias must align (12h and 1d HMA direction)

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 4h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_crsi_hma_donchian_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines momentum, streak, and percentile rank
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # RSI on streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive for RSI calculation
    streak_abs = np.abs(streak)
    streak_rsi = calculate_rsi(streak_abs, streak_period)
    
    # Percentile rank of close over last 100 bars
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    crsi = (rsi_close + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    > 61.8 = ranging, < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = np.zeros(n)
    atr_sum[:] = np.nan
    for i in range(period, n):
        atr_sum[i] = np.sum(tr[i-period+1:i+1])
    
    # Highest high and lowest low over period
    hh = np.zeros(n)
    ll = np.zeros(n)
    hh[:] = np.nan
    ll[:] = np.nan
    for i in range(period, n):
        hh[i] = np.max(high[i-period+1:i+1])
        ll[i] = np.min(low[i-period+1:i+1])
    
    # Calculate CHOP
    chop = np.zeros(n)
    chop[:] = np.nan
    for i in range(period, n):
        if hh[i] > ll[i] and atr_sum[i] > 0:
            chop[i] = 100.0 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
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
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    hma_21 = calculate_hma(close, period=21)
    hma_63 = calculate_hma(close, period=63)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
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
        
        if np.isnan(hma_21[i]) or np.isnan(hma_63[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
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
        
        # === HTF BIAS (12h and 1d HMA) ===
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong HTF alignment
        htf_strong_bull = htf_12h_bull and htf_1d_bull
        htf_strong_bear = htf_12h_bear and htf_1d_bear
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 61.8
        is_trend = chop[i] < 38.2
        
        # === HMA CROSSOVER TREND ===
        hma_bull = hma_21[i] > hma_63[i]
        hma_bear = hma_21[i] < hma_63[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakdown_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === CONNORS RSI EXTREMES (for range regime) ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion with Connors RSI
        if is_range:
            # Long: HTF bull + CRSI oversold
            if htf_strong_bull and crsi_oversold:
                desired_signal = SIZE_STRONG
            elif htf_12h_bull and crsi_oversold:
                desired_signal = SIZE_BASE
            # Short: HTF bear + CRSI overbought
            elif htf_strong_bear and crsi_overbought:
                desired_signal = -SIZE_STRONG
            elif htf_12h_bear and crsi_overbought:
                desired_signal = -SIZE_BASE
        
        # TREND REGIME: Trend following with HMA + Donchian
        elif is_trend:
            # Long: HTF bull + HMA bull + Donchian breakout
            if htf_strong_bull and hma_bull and donchian_breakout_long:
                desired_signal = SIZE_STRONG
            elif htf_12h_bull and hma_bull and donchian_breakout_long:
                desired_signal = SIZE_BASE
            # Short: HTF bear + HMA bear + Donchian breakdown
            elif htf_strong_bear and hma_bear and donchian_breakdown_short:
                desired_signal = -SIZE_STRONG
            elif htf_12h_bear and hma_bear and donchian_breakdown_short:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME (38.2 <= CHOP <= 61.8): Use HMA crossover only
        else:
            if htf_strong_bull and hma_bull:
                desired_signal = SIZE_BASE
            elif htf_strong_bear and hma_bear:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
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