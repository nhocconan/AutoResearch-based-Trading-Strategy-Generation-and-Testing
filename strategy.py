#!/usr/bin/env python3
"""
Experiment #1362: 12h Primary + 1d/1w HTF — Dual Regime with Connors RSI

Hypothesis: #1352 (12h) achieved Sharpe=0.571 with simple trend following. #1354 (4h) 
failed with Sharpe=-0.054. Pattern: HIGHER timeframes + SIMPLER logic = better results.

This strategy combines:
1. 12h HMA(21) for primary trend bias (proven in #1352)
2. 1d HMA(21) for intermediate confirmation
3. 1w HMA(21) for macro trend (NEW - adds longer-term filter)
4. Connors RSI (CRSI) for mean reversion entries within trend
5. Choppiness Index as SOFT regime filter (not hard requirement)
6. Asymmetric position sizing (0.30 with trend, 0.20 counter-trend)
7. ATR(14) trailing stop 2.0x (tighter than #1354's 2.5x)

Key differences from failed #1354:
- 12h primary (proven) vs 4h (failed)
- Connors RSI instead of standard RSI (better for mean reversion)
- 1w HMA for macro filter (adds longer-term confirmation)
- Choppiness as soft filter only (not hard requirement)
- Multiple entry paths to ensure >=30 trades/train

Target: 25-45 trades/year, Sharpe > 0.618, trades >= 30 train, >= 3 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_hma_1d1w_chop_atr_dual_regime_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator with 75% win rate
    """
    n = len(close)
    if n < rank_period + rsi_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3) - short term momentum
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak - consecutive up/down days
    streak_rsi = np.full(n, np.nan)
    streak = np.zeros(n, dtype=int)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_mask = streak_loss_smooth > 1e-10
    streak_rsi[streak_mask] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[streak_mask] / streak_loss_smooth[streak_mask]))
    streak_rsi[streak_loss_smooth <= 1e-10] = 100.0
    streak_rsi[:streak_period] = np.nan
    
    # Percent Rank - where current return ranks vs last 100 bars
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0 and not np.any(np.isnan(returns)):
            current_return = returns[-1]
            rank = np.sum(returns < current_return) / len(returns)
            pct_rank[i] = rank * 100.0
    
    # Combine into CRSI
    crsi = np.full(n, np.nan)
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(pct_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + pct_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures if market is trending or ranging
    CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        if highest > lowest:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

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
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend filters
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
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
            continue
        if np.isnan(hma_12h[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1w HMA) - strongest filter ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA) ===
        inter_bull = close[i] > hma_1d_aligned[i]
        inter_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h HMA) ===
        trend_bull = close[i] > hma_12h[i]
        trend_bear = close[i] < hma_12h[i]
        
        # === CHOPPINESS REGIME (SOFT filter) ===
        is_choppy = chop[i] > 55.0  # Soft threshold for range
        is_trending = chop[i] < 45.0  # Soft threshold for trend
        
        # === CONNORS RSI (Mean Reversion Signals) ===
        crsi_oversold = crsi[i] < 15.0  # Strong buy signal
        crsi_overbought = crsi[i] > 85.0  # Strong sell signal
        crsi_neutral_low = crsi[i] < 30.0
        crsi_neutral_high = crsi[i] > 70.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Multiple paths to ensure trades happen
        # Path 1: CRSI oversold + macro bull (mean reversion in uptrend) - PRIMARY
        if crsi_oversold and macro_bull:
            desired_signal = BASE_SIZE
        # Path 2: All HMAs aligned bull + CRSI neutral low (trend continuation)
        elif trend_bull and inter_bull and macro_bull and crsi_neutral_low:
            desired_signal = BASE_SIZE
        # Path 3: Choppiness low (trending) + price above all HMAs
        elif is_trending and trend_bull and inter_bull:
            desired_signal = REDUCED_SIZE
        # Path 4: Simple trend follow (price above 12h HMA + CRSI > 50)
        elif trend_bull and crsi[i] > 50.0:
            desired_signal = REDUCED_SIZE * 0.5
        
        # SHORT ENTRY: Multiple paths to ensure trades happen
        # Path 1: CRSI overbought + macro bear (mean reversion in downtrend) - PRIMARY
        elif crsi_overbought and macro_bear:
            desired_signal = -BASE_SIZE
        # Path 2: All HMAs aligned bear + CRSI neutral high (trend continuation)
        elif trend_bear and inter_bear and macro_bear and crsi_neutral_high:
            desired_signal = -BASE_SIZE
        # Path 3: Choppiness low (trending) + price below all HMAs
        elif is_trending and trend_bear and inter_bear:
            desired_signal = -REDUCED_SIZE
        # Path 4: Simple trend follow (price below 12h HMA + CRSI < 50)
        elif trend_bear and crsi[i] < 50.0:
            desired_signal = -REDUCED_SIZE * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
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
        if desired_signal > 0.15:
            final_signal = BASE_SIZE
        elif desired_signal > 0.05:
            final_signal = REDUCED_SIZE
        elif desired_signal < -0.15:
            final_signal = -BASE_SIZE
        elif desired_signal < -0.05:
            final_signal = -REDUCED_SIZE
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