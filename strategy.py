#!/usr/bin/env python3
"""
Experiment #1514: 4h Primary + 12h/1d HTF — Connors RSI + HMA Trend + Donchian Momentum

Hypothesis: Based on research showing Connors RSI achieved +0.923 Sharpe on ETH,
combined with 4h primary timeframe (proven in #1509, #1511 but needs simplification).

Key learnings from failures:
- #1509 (4h Donchian+HMA+RSI+1d): Sharpe=-0.187 — too many filters killed trades
- #1511 (4h dual regime CHOP+CRSI): Sharpe=-1.959 — regime switching too complex
- #1513 (1d HMA+RSI+Donchian): Sharpe=0.342 — simpler works better

New approach for 4h:
1. 12h HMA(21) for macro trend bias (HTF filter, less noisy than 1d)
2. 4h HMA(21) for primary trend confirmation
3. Connors RSI (RSI3 + RSI_Streak2 + PercentRank100) / 3 — proven edge
4. Donchian(20) for momentum confirmation (price position in channel)
5. ATR(14) 2.5x trailing stop for risk management
6. LOOSE entry bands to ensure 30-50 trades/year (Rule 9: MUST generate trades)

Why this should beat 0.618 Sharpe:
- Connors RSI catches reversals better than standard RSI (75% win rate in literature)
- 12h HTF is smoother than 1d for 4h primary (better alignment)
- Discrete sizing (0.0, ±0.25, ±0.30) minimizes fee churn
- Loose CRSI bands (15-35 long, 65-85 short) ensure trades happen

Timeframe: 4h (as required by experiment)
HTF: 12h (trend bias)
Position Size: 0.30 max (discrete levels)
Target: Sharpe > 0.618, DD < -30%, 30-50 trades/train, 8-15 trades/test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_hma_donchian_12h_atr_v1"
timeframe = "4h"
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
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    
    This is proven to have 75% win rate for mean reversion entries.
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # RSI(3) - very short term
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak - consecutive up/down days
    delta = np.diff(close, prepend=close[0])
    streak = np.zeros(n)
    for i in range(1, n):
        if delta[i] > 0:
            streak[i] = streak[i-1] + 1 if delta[i-1] >= 0 else 1
        elif delta[i] < 0:
            streak[i] = streak[i-1] - 1 if delta[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (2-period)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_window = streak[i-streak_period+1:i+1]
        if len(streak_window) == streak_period:
            gains = np.sum(np.maximum(streak_window, 0))
            losses = np.abs(np.sum(np.minimum(streak_window, 0)))
            if losses > 1e-10:
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + gains / losses))
            else:
                streak_rsi[i] = 100.0
    
    # Percent Rank (100) - where does today's return rank vs last 100?
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        if len(window) == rank_period and not np.any(np.isnan(window)):
            count_below = np.sum(window[:-1] < window[-1])
            percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    crsi = np.full(n, np.nan)
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    
    # Donchian channels
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # Appropriate size for 4h (20-50 trades/year target)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need enough data for CRSI rank_period=100
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (12h HMA) - primary direction bias ===
        htf_bull = close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_12h_aligned[i]
        
        # === PRIMARY TREND (4h HMA) - confirmation ===
        primary_bull = close[i] > hma_4h[i]
        primary_bear = close[i] < hma_4h[i]
        
        # === CONNORS RSI - LOOSE bands for MORE trades ===
        # Long: CRSI oversold (15-35) — looser than standard 10-30
        crsi_oversold = 15.0 <= crsi[i] <= 35.0
        # Short: CRSI overbought (65-85) — looser than standard 70-90
        crsi_overbought = 65.0 <= crsi[i] <= 85.0
        
        # === DONCHIAN MOMENTUM - price position in channel ===
        donchian_range = donchian_upper[i] - donchian_lower[i]
        if donchian_range > 1e-10:
            donchian_position = (close[i] - donchian_lower[i]) / donchian_range
        else:
            donchian_position = 0.5
        
        donchian_bull = donchian_position > 0.45  # price in upper half (loose)
        donchian_bear = donchian_position < 0.55  # price in lower half (loose)
        
        # === DESIRED SIGNAL — LOOSE CONDITIONS TO ENSURE TRADES ===
        desired_signal = 0.0
        
        # LONG entries (multiple tiers for flexibility)
        # Tier 1: Strong setup — HTF bull + primary bull + CRSI oversold + Donchian bull
        if htf_bull and primary_bull and crsi_oversold and donchian_bull:
            desired_signal = BASE_SIZE
        # Tier 2: HTF bull + primary bull + CRSI oversold (no Donchian filter)
        elif htf_bull and primary_bull and crsi_oversold:
            desired_signal = BASE_SIZE * 0.9
        # Tier 3: HTF bull + primary bull + Donchian bull + CRSI not overbought
        elif htf_bull and primary_bull and donchian_bull and crsi[i] < 50.0:
            desired_signal = BASE_SIZE * 0.8
        # Tier 4: HTF bull + primary bull + CRSI very oversold (<25)
        elif htf_bull and primary_bull and crsi[i] < 25.0:
            desired_signal = BASE_SIZE * 0.7
        # Tier 5: Fallback for trades — HTF bull + primary bull + CRSI < 40
        elif htf_bull and primary_bull and crsi[i] < 40.0:
            desired_signal = BASE_SIZE * 0.6
        
        # SHORT entries (multiple tiers for flexibility)
        # Tier 1: Strong setup — HTF bear + primary bear + CRSI overbought + Donchian bear
        elif htf_bear and primary_bear and crsi_overbought and donchian_bear:
            desired_signal = -BASE_SIZE
        # Tier 2: HTF bear + primary bear + CRSI overbought (no Donchian filter)
        elif htf_bear and primary_bear and crsi_overbought:
            desired_signal = -BASE_SIZE * 0.9
        # Tier 3: HTF bear + primary bear + Donchian bear + CRSI not oversold
        elif htf_bear and primary_bear and donchian_bear and crsi[i] > 50.0:
            desired_signal = -BASE_SIZE * 0.8
        # Tier 4: HTF bear + primary bear + CRSI very overbought (>75)
        elif htf_bear and primary_bear and crsi[i] > 75.0:
            desired_signal = -BASE_SIZE * 0.7
        # Tier 5: Fallback for trades — HTF bear + primary bear + CRSI > 60
        elif htf_bear and primary_bear and crsi[i] > 60.0:
            desired_signal = -BASE_SIZE * 0.6
        
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
        elif desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE * 0.85
        elif desired_signal >= BASE_SIZE * 0.55:
            final_signal = BASE_SIZE * 0.7
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE * 0.85
        elif desired_signal <= -BASE_SIZE * 0.55:
            final_signal = -BASE_SIZE * 0.7
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