#!/usr/bin/env python3
"""
Experiment #1650: 1h Primary + 4h/12h HTF — Connors RSI Mean Reversion + Choppiness Regime

Hypothesis: Previous 1h strategies failed due to OVER-FILTERING (0 trades). This strategy
uses a SCORING SYSTEM instead of hard AND filters, allowing partial confluence to trigger
entries. Key innovations:

1. Connors RSI (CRSI): Proven 75% win rate mean reversion signal
   - RSI(3) + RSI_Streak(2) + PercentRank(100) / 3
   - Long: CRSI < 20, Short: CRSI > 80 (looser than typical 10/90)

2. Choppiness Index regime: CHOP(14) > 55 = range (mean revert), CHOP < 45 = trend
   - In range: use CRSI extremes for mean reversion
   - In trend: use CRSI pullbacks in trend direction

3. 4h HMA(21) bias: Only trade long if 4h bullish, short if 4h bearish
   - Provides HTF direction without over-filtering

4. Session filter: Only 8-20 UTC (high volume hours) - but NOT required for all trades

5. Scoring system: Each factor adds points, enter when score > threshold
   - This prevents 0 trades from over-filtering

6. ATR(14) 2.5x trailing stoploss: Controlled drawdown

Timeframe: 1h (required for this experiment)
HTF: 4h HMA via mtf_data.get_htf_data() — called ONCE before loop
Target: 40-80 trades/year, Sharpe > 0.5, DD > -35%
Size: 0.20 (smaller for 1h due to higher fee sensitivity)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_4h_score_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors Relative Strength Index (CRSI)
    Combines 3 components for mean reversion signal
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(np.concatenate([[0], gain])).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(np.concatenate([[0], loss])).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if loss_smooth[i] < 1e-10:
            rsi_short[i] = 100.0
        else:
            rsi_short[i] = 100.0 - (100.0 / (1.0 + gain_smooth[i] / loss_smooth[i]))
    
    # Component 2: RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        abs_streak = min(abs(streak[i]), streak_period)
        if streak[i] >= 0:
            streak_rsi[i] = 50.0 + (abs_streak / streak_period) * 50.0
        else:
            streak_rsi[i] = 50.0 - (abs_streak / streak_period) * 50.0
    
    # Component 3: Percent Rank (where close sits in last N bars)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        if len(window) < rank_period:
            continue
        current = close[i]
        rank = np.sum(window[:-1] < current)
        percent_rank[i] = (rank / (rank_period - 1)) * 100.0
    
    # Combine components
    crsi = np.full(n, np.nan)
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppy vs trending
    CHOP > 61.8 = range, CHOP < 38.2 = trend
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest or highest - lowest < 1e-10:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

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

def calculate_volume_avg(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_avg[i] = np.mean(volume[i-period+1:i+1])
    
    return vol_avg

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.20  # Smaller size for 1h due to fee sensitivity
    
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
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SCORING SYSTEM (not hard AND filters) ===
        long_score = 0.0
        short_score = 0.0
        
        # 4h HMA trend bias (strong weight)
        if close[i] > hma_4h_aligned[i]:
            long_score += 40  # 4h bullish favors longs
        else:
            short_score += 40  # 4h bearish favors shorts
        
        # Connors RSI extremes (mean reversion signal)
        if crsi[i] < 20:
            long_score += 35  # Oversold
        elif crsi[i] < 35:
            long_score += 15  # Moderately oversold
        
        if crsi[i] > 80:
            short_score += 35  # Overbought
        elif crsi[i] > 65:
            short_score += 15  # Moderately overbought
        
        # Choppiness regime
        if chop[i] > 55:  # Range market - mean reversion favored
            if crsi[i] < 30:
                long_score += 20
            if crsi[i] > 70:
                short_score += 20
        else:  # Trending market - follow 4h bias
            if close[i] > hma_4h_aligned[i] and crsi[i] < 50:
                long_score += 15  # Pullback in uptrend
            if close[i] < hma_4h_aligned[i] and crsi[i] > 50:
                short_score += 15  # Pullback in downtrend
        
        # Volume confirmation (optional bonus)
        if not np.isnan(vol_avg[i]) and vol_avg[i] > 0:
            if volume[i] > 0.8 * vol_avg[i]:
                long_score += 5
                short_score += 5
        
        # Session filter (8-20 UTC) - bonus points only
        try:
            hour = pd.to_datetime(open_time[i], unit='ms').hour
            if 8 <= hour <= 20:
                long_score += 5
                short_score += 5
        except:
            pass
        
        # === DECISION THRESHOLD ===
        desired_signal = 0.0
        ENTRY_THRESHOLD = 65  # Need score > 65 to enter
        
        if long_score >= ENTRY_THRESHOLD and long_score > short_score:
            desired_signal = BASE_SIZE
        elif short_score >= ENTRY_THRESHOLD and short_score > long_score:
            desired_signal = -BASE_SIZE
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
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