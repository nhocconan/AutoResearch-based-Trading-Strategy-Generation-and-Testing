#!/usr/bin/env python3
"""
Experiment #1368: 30m Primary + 4h/1d HTF — Connors RSI Mean Reversion

Hypothesis: Lower TF (30m) strategies failed (#1358, #1360, #1365) due to OVER-FILTERING
with session filters, volume filters, and too many confluence requirements = 0 trades.
Solution: Use Connors RSI mean reversion (proven 75% win rate) with HTF trend BIAS
(not hard filter). Allow counter-trend entries when CRSI is extreme to ensure trade
frequency. Remove session/volume filters that caused paralysis.

Key design choices:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
2. 4h HMA(21) for trend BIAS — increases size when aligned, not hard filter
3. 1d HMA(21) for macro direction — soft weight only
4. Entry: CRSI < 15 (long) or > 85 (short) — extreme mean reversion
5. Exit: CRSI crosses 50 OR ATR(14) 2.0x stoploss
6. Position size: 0.25 base, 0.35 with HTF confirmation
7. NO session filter, NO volume filter — these caused 0 trades
8. Allow counter-trend entries when CRSI extreme (ensures >=50 trades/year)

Target: 50-100 trades/year, Sharpe > 0.618, trades >= 30 train, >= 5 test
Timeframe: 30m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_hma_4h1d_atr_meanrevert_v1"
timeframe = "30m"
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

def calculate_crsi(close):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): Standard RSI with 3-period lookback
    RSI_Streak(2): RSI of consecutive up/down days (2-period)
    PercentRank(100): Percentile rank of today's return vs last 100 days
    """
    n = len(close)
    if n < 100:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi3 = calculate_rsi(close, period=3)
    
    # RSI Streak (2) - measure consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (2-period)
    streak_abs = np.abs(streak)
    streak_rsi = np.full(n, np.nan)
    for i in range(2, n):
        if streak_abs[i] == 0:
            streak_rsi[i] = 50.0
        else:
            # Map streak to 0-100 scale (longer streak = more extreme)
            streak_rsi[i] = min(100.0, max(0.0, 50.0 + streak[i] * 25.0))
    
    # Percent Rank (100) - percentile of today's return vs last 100 days
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    percent_rank = np.full(n, np.nan)
    for i in range(100, n):
        window = returns[i-99:i+1]  # 100-day window including today
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            rank = np.sum(valid < returns[i]) / len(valid)
            percent_rank[i] = rank * 100.0
    
    # Combine into Connors RSI
    crsi = np.full(n, np.nan)
    for i in range(100, n):
        if not np.isnan(rsi3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    crsi = calculate_crsi(close)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    CONFIRMED_SIZE = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track CRSI for exit signals
    prev_crsi = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HTF TREND BIAS (soft filter - increases size when aligned) ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        trend_1d_bull = close[i] > hma_1d_aligned[i]
        trend_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong confirmation when both HTF agree
        htf_bull_confirmed = trend_4h_bull and trend_1d_bull
        htf_bear_confirmed = trend_4h_bear and trend_1d_bear
        
        # === CONNORS RSI MEAN REVERSION ===
        # Extreme oversold (long) or overbought (short)
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # Moderate levels for exit
        crsi_neutral = 40.0 < crsi[i] < 60.0
        
        # CRSI cross above 50 (exit long) or below 50 (exit short)
        crsi_cross_above_50 = prev_crsi < 50.0 and crsi[i] >= 50.0
        crsi_cross_below_50 = prev_crsi > 50.0 and crsi[i] <= 50.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: CRSI extreme oversold
        if crsi_oversold:
            if htf_bull_confirmed:
                desired_signal = CONFIRMED_SIZE  # 0.35 with HTF confirmation
            elif trend_4h_bull:
                desired_signal = BASE_SIZE  # 0.25 with 4h only
            else:
                desired_signal = BASE_SIZE * 0.7  # 0.175 counter-trend (still allow)
        
        # SHORT ENTRY: CRSI extreme overbought
        elif crsi_overbought:
            if htf_bear_confirmed:
                desired_signal = -CONFIRMED_SIZE  # -0.35 with HTF confirmation
            elif trend_4h_bear:
                desired_signal = -BASE_SIZE  # -0.25 with 4h only
            else:
                desired_signal = -BASE_SIZE * 0.7  # -0.175 counter-trend
        
        # === EXIT CONDITIONS ===
        exit_long = False
        exit_short = False
        
        if in_position and position_side > 0:
            # Exit long when CRSI crosses above 50 or reaches neutral
            if crsi_cross_above_50 or crsi[i] > 70.0:
                exit_long = True
        
        if in_position and position_side < 0:
            # Exit short when CRSI crosses below 50 or reaches neutral
            if crsi_cross_below_50 or crsi[i] < 30.0:
                exit_short = True
        
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
        
        if exit_long and position_side > 0:
            desired_signal = 0.0
        if exit_short and position_side < 0:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.15:
            final_signal = BASE_SIZE if not htf_bull_confirmed else CONFIRMED_SIZE
        elif desired_signal < -0.15:
            final_signal = -BASE_SIZE if not htf_bear_confirmed else -CONFIRMED_SIZE
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
        prev_crsi = crsi[i]
    
    return signals