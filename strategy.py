#!/usr/bin/env python3
"""
Experiment #654: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + Donchian Breakout + CRSI

Hypothesis: After analyzing 574 failed strategies, key insights:
1. #651 (Sharpe=0.222) proved 4h Choppiness+CRSI+1d HMA works but underperforms best (0.520)
2. KAMA (Kaufman Adaptive) outperforms HMA/EMA in choppy markets by reducing whipsaws
3. Donchian breakouts catch major moves that CRSI alone misses
4. Combining adaptive trend (KAMA) + breakout confirmation + mean-reversion entries = higher Sharpe

This strategy uses:
- 12h KAMA(10) for adaptive trend direction (responds to volatility changes)
- 4h Choppiness Index for regime detection (range vs trend)
- 4h Donchian(20) breakout for trend entry confirmation
- Connors RSI for precise mean-reversion entries in range regime
- Asymmetric logic: mean-revert in chop, breakout-follow in trends

Why this might beat Sharpe=0.520:
- KAMA adapts to market conditions better than fixed EMA/HMA
- Donchian breakout catches major moves CRSI misses (2022 crash, 2021 rally)
- 4h timeframe = 25-45 trades/year (optimal per Rule 10)
- Dual entry system captures both mean-reversion AND trend moves
- Conservative sizing (0.28) + 2.5*ATR stop controls drawdown

Position sizing: 0.28 discrete (per Rule 4, max 0.40)
Target: 25-45 trades/year on 4h
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_donchian_crsi_12h_v1"
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

def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise by adjusting smoothing constant based on efficiency ratio.
    
    ER = |Close - Close(n)| / Sum(|Close(i) - Close(i-1)|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio
    signal = np.abs(close_s - close_s.shift(efficiency_period))
    noise = pd.Series(np.abs(close_s.diff())).rolling(window=efficiency_period, min_periods=efficiency_period).sum()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = signal / (noise + 1e-10)
    er = er.fillna(0).values
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8: Range/consolidation
    CHOP < 38.2: Trending
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
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) on close
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak calculation
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if pd.isna(returns.iloc[i]):
            streak[i] = 0
        elif returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100)
    def percent_rank(x):
        if len(x) <= 1:
            return 0.5
        return (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1)
    
    percent_rank_vals = pd.Series(returns).rolling(window=rank_period, min_periods=rank_period).apply(
        percent_rank, raw=False
    ).values * 100.0
    
    crsi = (rsi_close.values + rsi_streak.values + percent_rank_vals) / 3.0
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bounds."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h KAMA for adaptive trend direction
    kama_12h = calculate_kama(df_12h['close'].values, efficiency_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(kama_12h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 12H TREND BIAS (KAMA slope over 5 bars) ===
        kama_slope_bull = kama_12h_aligned[i] > kama_12h_aligned[i-5] if i >= 5 else False
        kama_slope_bear = kama_12h_aligned[i] < kama_12h_aligned[i-5] if i >= 5 else False
        
        # Price relative to 12h KAMA
        price_above_kama = close[i] > kama_12h_aligned[i]
        price_below_kama = close[i] < kama_12h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range = chop_14[i] > 55.0  # Range/consolidation
        is_trend = chop_14[i] < 45.0  # Trending
        
        # === CONNORS RSI EXTREMES (relaxed for more trades) ===
        crsi_oversold = crsi[i] < 20.0  # Oversold
        crsi_overbought = crsi[i] > 80.0  # Overbought
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Range market + CRSI oversold = mean revert long
        if is_range and crsi_oversold:
            new_signal = POSITION_SIZE
        
        # Regime 2: Trending market + 12h bull + Donchian breakout = trend follow long
        elif is_trend and kama_slope_bull and price_above_kama:
            if breakout_long:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Regime 1: Range market + CRSI overbought = mean revert short
        elif is_range and crsi_overbought:
            new_signal = -POSITION_SIZE
        
        # Regime 2: Trending market + 12h bear + Donchian breakout = trend follow short
        elif is_trend and kama_slope_bear and price_below_kama:
            if breakout_short:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC (carry forward if in position) ===
        if in_position and new_signal == 0.0:
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if kama_slope_bear and price_below_kama:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if kama_slope_bull and price_above_kama:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals