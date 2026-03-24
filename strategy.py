#!/usr/bin/env python3
"""
Experiment #490: 1h Primary + 4h/1d HTF — Choppiness + cRSI + Session Filter

Hypothesis: 1h timeframe with strict regime-based confluence filters will generate 40-80 trades/year
while maintaining positive Sharpe across BTC/ETH/SOL. Key innovations:
1. 1d HMA(21) = overall trend bias for trend-following trades
2. 4h HMA(21) = intermediate trend confirmation
3. Choppiness Index(14) = regime filter (>61.8 range, <38.2 trend)
4. Connors RSI = entry trigger (extreme <15 long, >85 short)
5. Session filter 08-20 UTC = avoid low liquidity whipsaws
6. ATR(14)*2.0 stoploss on all positions

Why this should work:
- Session filter alone cuts 66% of potential entries (reduces fee drag)
- Choppiness ensures we only trade in appropriate regime (trend vs mean-revert)
- cRSI extreme values are rare (5-10% of bars) = fewer trades
- HTF alignment prevents counter-trend trades in trending regime
- Range regime allows counter-trend trades (mean reversion)
- Target: 50-80 trades/year on 1h = 1-2 trades per week

Position sizing: 0.20 base, 0.30 strong signals (discrete levels to minimize churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_session_4h1d_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean-reversion indicator with 75% win rate at extremes
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        streak_window = streak[max(0, i-streak_period+1):i+1]
        if len(streak_window) > 0:
            pos_count = np.sum(streak_window > 0)
            neg_count = np.sum(streak_window < 0)
            if pos_count + neg_count > 0:
                streak_rsi[i] = (pos_count / (pos_count + neg_count)) * 100.0
            else:
                streak_rsi[i] = 50.0
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        if len(window) > 1:
            rank = np.sum(window[:-1] < close[i])
            percent_rank[i] = (rank / (len(window) - 1)) * 100.0
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    valid_mask = (~np.isnan(rsi_short)) & (~np.isnan(streak_rsi)) & (~np.isnan(percent_rank))
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures if market is trending or ranging
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high - lowest_low > 1e-10 and sum_tr > 1e-10:
            choppiness[i] = 100.0 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(period)
    
    return choppiness

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h HMA and Choppiness
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    chop_4h_raw = calculate_choppiness(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    # Calculate 1h indicators
    hma_1h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_1h[i]) or np.isnan(chop_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # Convert open_time to hour (open_time is in milliseconds)
        hour = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour <= 20
        
        # === 1d HTF BIAS ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HTF TREND ===
        htf4_bull = close[i] > hma_4h_aligned[i]
        htf4_bear = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME (use 4h for regime, more stable) ===
        chop_val = chop_4h_aligned[i]
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8
        
        # === CRSI EXTREMES ===
        crsi_extreme_low = crsi[i] < 15.0
        crsi_extreme_high = crsi[i] > 85.0
        crsi_super_low = crsi[i] < 10.0
        crsi_super_high = crsi[i] > 90.0
        
        # === ENTRY LOGIC (REGIME-BASED) ===
        desired_signal = 0.0
        
        # TREND-FOLLOWING LONG: trending regime + HTF bull + CRSI pullback
        if in_session and is_trending:
            if htf_bull and htf4_bull and crsi_extreme_low:
                desired_signal = SIZE_STRONG
            elif htf_bull and crsi[i] < 35.0 and crsi[i-1] < 35.0:
                # CRSI staying low in uptrend = pullback entry
                desired_signal = SIZE_BASE
        
        # TREND-FOLLOWING SHORT: trending regime + HTF bear + CRSI rally
        elif in_session and is_trending:
            if htf_bear and htf4_bear and crsi_extreme_high:
                desired_signal = -SIZE_STRONG
            elif htf_bear and crsi[i] > 65.0 and crsi[i-1] > 65.0:
                # CRSI staying high in downtrend = rally entry
                desired_signal = -SIZE_BASE
        
        # MEAN REVERSION LONG: ranging regime + CRSI extreme (no HTF filter)
        if desired_signal == 0.0 and in_session and is_ranging:
            if crsi_super_low:
                desired_signal = SIZE_STRONG
            elif crsi_extreme_low:
                desired_signal = SIZE_BASE
        
        # MEAN REVERSION SHORT: ranging regime + CRSI extreme (no HTF filter)
        if desired_signal == 0.0 and in_session and is_ranging:
            if crsi_super_high:
                desired_signal = -SIZE_STRONG
            elif crsi_extreme_high:
                desired_signal = -SIZE_BASE
        
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