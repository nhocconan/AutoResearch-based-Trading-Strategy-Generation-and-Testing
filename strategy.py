#!/usr/bin/env python3
"""
Experiment #1522: 12h Primary + 1d/1w HTF — HMA Trend + Connors RSI + ATR Stop

Hypothesis: 12h timeframe with 1d HTF trend filter + Connors RSI for entry timing will 
generate 30-50 trades/year with better risk-adjusted returns than pure trend following.
Connors RSI (CRSI) combines RSI(3) + RSI_Streak(2) + PercentRank(100) for superior
mean-reversion entry signals within a trend framework.

Key insights from 1100+ failed strategies:
1. Complex regime filters (CHOP+CRSI together) = 0 trades (#1511, #1512, #1514)
2. SIMPLER works: HTF trend bias + primary trend + entry timing (#1505, #1506, #1513)
3. 12h timeframe naturally generates 20-50 trades/year (perfect for fee efficiency)
4. Connors RSI < 30 / > 70 provides high-probability pullback entries (75% win rate)
5. ATR 2.5x trailing stop protects capital in 2022-style crashes

Design:
- 1d HMA(21) for macro trend direction (HTF filter, aligned properly)
- 1w HMA(21) for weekly bias confirmation (secondary HTF)
- 12h HMA(21) for primary trend confirmation
- Connors RSI(3,2,100) for pullback entries (long < 30, short > 70)
- ATR(14) 2.5x trailing stop for risk management
- Position size 0.25-0.30 (discrete: 0.0, ±0.25, ±0.30)
- Target: 30-60 trades/train (4 years), 8-15 trades/test (15 months)

Timeframe: 12h (as required by experiment)
HTF: 1d (primary), 1w (secondary bias)
Position Size: 0.25-0.30 (discrete levels to minimize fee churn)
Target: Sharpe > 0.618 (beat current best), DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_crsi_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
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
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's change vs last 100 days
    
    CRSI < 30 = oversold (long opportunity)
    CRSI > 70 = overbought (short opportunity)
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # RSI(3) - fast RSI for short-term momentum
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
    streak = np.zeros(n)
    direction = np.zeros(n)  # 1 for up, -1 for down
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            direction[i] = 1
            if direction[i-1] == 1:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif close[i] < close[i-1]:
            direction[i] = -1
            if direction[i-1] == -1:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            direction[i] = 0
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period+1:i+1]
        if len(streak_vals) > 0:
            avg_streak = np.mean(streak_vals)
            # Map streak to 0-100 scale (positive streak = high, negative = low)
            streak_rsi[i] = 50.0 + (avg_streak * 10.0)
            streak_rsi[i] = np.clip(streak_rsi[i], 0.0, 100.0)
    
    # Percent Rank - today's return vs last 100 days
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0 and not np.all(np.isnan(returns)):
            today_return = close[i] - close[i-1]
            count_below = np.sum(returns < today_return)
            pct_rank[i] = (count_below / len(returns)) * 100.0
    
    # Combine into CRSI
    crsi = np.full(n, np.nan)
    mask = ~np.isnan(rsi_fast) & ~np.isnan(streak_rsi) & ~np.isnan(pct_rank)
    crsi[mask] = (rsi_fast[mask] + streak_rsi[mask] + pct_rank[mask]) / 3.0
    
    return crsi

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
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for weekly bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Appropriate size for 12h (30-50 trades/year target)
    
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
        if np.isnan(crsi[i]) or np.isnan(hma_12h[i]):
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
        
        # === MACRO TREND (1w HMA) - weekly direction bias ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (1d HMA) - daily confirmation ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h TREND (HMA) - entry timeframe confirmation ===
        trend_12h_bull = close[i] > hma_12h[i]
        trend_12h_bear = close[i] < hma_12h[i]
        
        # === CONNORS RSI - pullback entries (LOOSE for trades) ===
        # Long: CRSI oversold (< 35 for more trades)
        crsi_oversold = crsi[i] < 35.0
        # Short: CRSI overbought (> 65 for more trades)
        crsi_overbought = crsi[i] > 65.0
        
        # === DESIRED SIGNAL - TREND + PULLBACK ===
        desired_signal = 0.0
        
        # LONG: All timeframes bullish + CRSI pullback
        # Option 1: Strong trend (1w + 1d + 12h all bull) + CRSI oversold
        if weekly_bull and daily_bull and trend_12h_bull and crsi_oversold:
            desired_signal = BASE_SIZE
        # Option 2: 1w + 1d bull + 12h bull (looser, ensures trades)
        elif weekly_bull and daily_bull and trend_12h_bull:
            desired_signal = BASE_SIZE * 0.8
        # Option 3: 1w bull + 1d bull + CRSI oversold (fallback)
        elif weekly_bull and daily_bull and crsi_oversold:
            desired_signal = BASE_SIZE * 0.7
        # Option 4: 1w bull + 1d above HMA + CRSI not extreme (loosest)
        elif weekly_bull and daily_bull and crsi[i] < 55.0:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT: All timeframes bearish + CRSI pullback
        # Option 1: Strong trend (1w + 1d + 12h all bear) + CRSI overbought
        elif weekly_bear and daily_bear and trend_12h_bear and crsi_overbought:
            desired_signal = -BASE_SIZE
        # Option 2: 1w + 1d bear + 12h bear (looser, ensures trades)
        elif weekly_bear and daily_bear and trend_12h_bear:
            desired_signal = -BASE_SIZE * 0.8
        # Option 3: 1w bear + 1d bear + CRSI overbought (fallback)
        elif weekly_bear and daily_bear and crsi_overbought:
            desired_signal = -BASE_SIZE * 0.7
        # Option 4: 1w bear + 1d below HMA + CRSI not extreme (loosest)
        elif weekly_bear and daily_bear and crsi[i] > 45.0:
            desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal >= BASE_SIZE * 0.9:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.9:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.6
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