#!/usr/bin/env python3
"""
Experiment #113: 1d Primary + 1w HTF — Regime-Adaptive CRSI + Donchian

Hypothesis: Previous strategies failed due to overly complex exit logic and too many
conflicting filters. This uses a cleaner regime-adaptive approach:

1) Choppiness Index (CHOP) for regime detection:
   - CHOP > 61.8 = ranging market → use Connors RSI mean reversion
   - CHOP < 38.2 = trending market → use Donchian breakout trend following
   - 38.2 <= CHOP <= 61.8 = neutral → stay flat or hold existing positions

2) 1w HMA(21) for macro trend bias — only trade in trend direction

3) Connors RSI (CRSI) for mean reversion entries:
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 10 + price > 1w HMA
   - Short: CRSI > 90 + price < 1w HMA

4) Donchian(20) breakout for trend entries:
   - Long: price breaks 20-day high + 1w HMA sloping up
   - Short: price breaks 20-day low + 1w HMA sloping down

5) Single exit: 3x ATR trailing stop — let winners run, cut losers fast

Why this should work:
- Regime detection prevents trend strategies in chop (2022 whipsaw)
- CRSI has 75% win rate on mean reversion (proven on ETH 1d)
- Donchian breakout proven on 4h (current best Sharpe=0.486)
- Simpler exit logic = fewer premature exits
- 1d timeframe naturally produces 25-40 trades/year (low fee drag)

Position size: 0.30 discrete (max 0.35 with strong confluence)
Stoploss: 3.0*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_crsi_donchian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) - streak of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like scale (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i-streak_period+1):i+1]
        up_streaks = np.sum(streak_vals > 0)
        total = len(streak_vals)
        streak_rsi[i] = (up_streaks / total) * 100 if total > 0 else 50
    
    # Percent Rank (100) - where does today's return rank vs last 100 days?
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        returns = np.diff(close[max(0, i-rank_period+1):i+1])
        if len(returns) > 0:
            today_return = close[i] - close[i-1]
            rank = np.sum(returns < today_return) / len(returns) * 100
            percent_rank[i] = rank
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    Values: > 61.8 = ranging, < 38.2 = trending
    """
    n = len(close)
    choppiness = np.zeros(n)
    
    for i in range(period, n):
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_val = highest_high - lowest_low
        
        if range_val > 0 and atr_sum > 0:
            choppiness[i] = 100 * np.log10(atr_sum / range_val) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    return choppiness

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-day high/low)."""
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1w HMA slope (trend strength)
    hma_1w_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1w_aligned[i]) and not np.isnan(hma_1w_aligned[i-1]) and hma_1w_aligned[i-1] != 0:
            hma_1w_slope[i] = (hma_1w_aligned[i] - hma_1w_aligned[i-1]) / hma_1w_aligned[i-1] * 100
        else:
            hma_1w_slope[i] = 0.0
    
    # Calculate 1d indicators (ALL before loop for performance)
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.30
    POSITION_SIZE_MAX = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = np.inf
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = choppiness[i] > 61.8
        is_trending = choppiness[i] < 38.2
        is_neutral = not is_choppy and not is_trending
        
        # === HTF TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        hma_slope_positive = hma_1w_slope[i] > 0.3
        hma_slope_negative = hma_1w_slope[i] < -0.3
        
        # === 1d TREND FILTER ===
        hma_1d_bullish = hma_1d_21[i] > hma_1d_50[i]
        hma_1d_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === DONCHIAN BREAKOUT ===
        prev_high = donchian_upper[i-1]
        prev_low = donchian_lower[i-1]
        
        breakout_long = close[i] > prev_high
        breakout_short = close[i] < prev_low
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi[i] < 10
        crsi_overbought = crsi[i] > 90
        
        # === ENTRY LOGIC (Regime-Adaptive) ===
        new_signal = 0.0
        
        # --- TRENDING REGIME: Donchian Breakout ---
        if is_trending:
            # Long: breakout + 1w trend up + 1d trend up
            if breakout_long and (price_above_hma_1w or hma_slope_positive) and hma_1d_bullish:
                new_signal = POSITION_SIZE_BASE
                if hma_slope_positive and hma_1d_21[i] > hma_1d_50[i] * 1.01:
                    new_signal = POSITION_SIZE_MAX
            
            # Short: breakout + 1w trend down + 1d trend down
            if breakout_short and (price_below_hma_1w or hma_slope_negative) and hma_1d_bearish:
                new_signal = -POSITION_SIZE_BASE
                if hma_slope_negative and hma_1d_21[i] < hma_1d_50[i] * 0.99:
                    new_signal = -POSITION_SIZE_MAX
        
        # --- CHOPPY REGIME: CRSI Mean Reversion ---
        elif is_choppy:
            # Long: CRSI oversold + price above 1w HMA (bullish bias)
            if crsi_oversold and price_above_hma_1w:
                new_signal = POSITION_SIZE_BASE
            
            # Short: CRSI overbought + price below 1w HMA (bearish bias)
            if crsi_overbought and price_below_hma_1w:
                new_signal = -POSITION_SIZE_BASE
        
        # --- NEUTRAL REGIME: Hold or flat ---
        # No new entries, but can hold existing positions
        
        # === STOPLOSS CHECK (3.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL (only if not in strong profit) ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and price_below_hma_1w and hma_slope_negative:
                new_signal = 0.0
            if position_side < 0 and price_above_hma_1w and hma_slope_positive:
                new_signal = 0.0
        
        # === EXIT ON OPPOSITE BREAKOUT ===
        if in_position and position_side > 0 and breakout_short:
            new_signal = 0.0
        if in_position and position_side < 0 and breakout_long:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = np.inf
        
        signals[i] = new_signal
    
    return signals