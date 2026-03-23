#!/usr/bin/env python3
"""
Experiment #154: 4h Primary + 12h/1d HTF — Dual Regime Strategy

Hypothesis: Single-regime strategies fail because they apply trend logic in chop
and mean-reversion logic in trends. This strategy uses Choppiness Index to
SWITCH between two distinct entry modes:

1) TREND REGIME (CHOP < 38.2): Donchian breakout + 12h HMA confirmation
   - Enter on 20-period Donchian breakout in trend direction
   - Only long if 12h HMA sloping up, only short if sloping down
   - Position size: 0.30

2) RANGE REGIME (CHOP > 61.8): CRSI mean reversion at extremes
   - Long when CRSI < 12 (oversold), short when CRSI > 88 (overbought)
   - Requires price near Bollinger bands for confluence
   - Position size: 0.25

3) TRANSITION REGIME (38.2 <= CHOP <= 61.8): Reduce position or flat
   - Only hold existing positions, no new entries
   - Prevents whipsaw during regime changes

Why 4h works:
- Proven timeframe (current best Sharpe=0.486 is 4h-based)
- 20-50 trades/year target (low fee drag)
- Captures multi-day swings without noise

HTF Filter: 12h HMA(21) for macro trend bias
- Long bias: 12h HMA sloping up (current > previous)
- Short bias: 12h HMA sloping down

Position sizing: 0.25 base, 0.35 max with confluence
Stoploss: 2.5*ATR trailing stop
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_crsi_donchian_12h_v1"
timeframe = "4h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.maximum(price_range, 1e-10)  # avoid div by zero
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-period high/low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - mean reversion indicator.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    CRSI < 15 = oversold (long), CRSI > 85 = overbought (short)
    """
    close_s = pd.Series(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI - consecutive up/down bars
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.clip(lower=0)
    streak_loss = (-streak_s).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank(100)
    returns = close_s.pct_change()
    percent_rank = returns.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100 if len(x) > 1 else 50,
        raw=False
    )
    percent_rank = percent_rank.fillna(50).values
    
    rsi_close_arr = rsi_close.fillna(50).values
    rsi_streak_arr = rsi_streak.fillna(50).values
    
    crsi = (rsi_close_arr + rsi_streak_arr + percent_rank) / 3.0
    return crsi

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for macro trend bias
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_dev=2.0)
    
    signals = np.zeros(n)
    POSITION_SIZE_TREND = 0.30
    POSITION_SIZE_RANGE = 0.25
    POSITION_SIZE_EXIT = 0.0
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_12h_aligned[i-1]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === HTF TREND BIAS (12h HMA slope) ===
        hma_12h_slope_up = hma_12h_aligned[i] > hma_12h_aligned[i-1]
        hma_12h_slope_down = hma_12h_aligned[i] < hma_12h_aligned[i-1]
        
        # === REGIME DETECTION ===
        is_trend_regime = chop[i] < 38.2
        is_range_regime = chop[i] > 61.8
        is_transition = (not is_trend_regime) and (not is_range_regime)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TREND REGIME: Donchian Breakout ---
        if is_trend_regime:
            # Long breakout with 12h HMA confirmation
            if hma_12h_slope_up and close[i] > donchian_upper[i-1]:
                new_signal = POSITION_SIZE_TREND
            
            # Short breakout with 12h HMA confirmation
            if hma_12h_slope_down and close[i] < donchian_lower[i-1]:
                new_signal = -POSITION_SIZE_TREND
        
        # --- RANGE REGIME: CRSI Mean Reversion ---
        if is_range_regime:
            # Long: CRSI oversold + price near lower BB
            if crsi[i] < 12.0 and close[i] <= bb_lower[i] * 1.005:
                new_signal = POSITION_SIZE_RANGE
            
            # Short: CRSI overbought + price near upper BB
            if crsi[i] > 88.0 and close[i] >= bb_upper[i] * 0.995:
                new_signal = -POSITION_SIZE_RANGE
        
        # --- TRANSITION REGIME: No new entries ---
        if is_transition:
            new_signal = 0.0
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and no exit signal (allow position to continue through regime changes)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if above Donchian mid
                if close[i] > donchian_mid[i]:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if below Donchian mid
                if close[i] < donchian_mid[i]:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TAKE PROFIT ON CRSI REVERSAL ===
        if in_position and position_side > 0 and crsi[i] > 80.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20.0:
            new_signal = 0.0
        
        # === EXIT ON MACRO TREND REVERSAL ===
        if in_position and position_side > 0:
            if hma_12h_slope_down and chop[i] < 38.2:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_12h_slope_up and chop[i] < 38.2:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals