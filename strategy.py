#!/usr/bin/env python3
"""
Experiment #664: 4h Primary + 12h HTF — Simplified Choppiness Regime + Connors RSI + ADX

Hypothesis: After analyzing 582+ failed strategies, the pattern is clear:
1. #651 (current) has too many filters = missed opportunities in fast moves
2. CRSI thresholds (15/85) are too extreme = too few trades
3. Dual choppiness threshold (45/55) creates no-man's-land where no signals fire
4. 1d HMA is too slow for 4h entries — 12h is better match

This strategy SIMPLIFIES the entry logic:
- Single choppiness threshold (50) to switch between mean-revert and trend-follow
- CRSI thresholds relaxed to 20/80 (still extreme but more actionable)
- ADX(14)>25 confirms trending regime (literature standard)
- 12h HMA for trend bias (faster than 1d, still HTF)
- Asymmetric entries: long bias in bull, short bias in bear

Why this might beat Sharpe=0.520:
- Fewer filters = more trades (target 35-50/year on 4h)
- CRSI 20/80 still captures 70%+ win rate extremes
- ADX filter prevents trend-following in weak trends (major loss source)
- 12h HMA responds faster to regime changes than 1d
- Conservative sizing (0.28) + ATR stop controls drawdown

Position sizing: 0.28 discrete (per Rule 4, max 0.40)
Target: 35-50 trades/year on 4h
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_crsi_adx_12h_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 50: Range/consolidation (mean-revert)
    CHOP < 50: Trending (trend-follow)
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

def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index).
    ADX > 25: Strong trend
    ADX < 20: Range/weak trend
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = tr1.combine(tr2, np.maximum).combine(tr3, np.maximum)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if pd.isna(returns.iloc[i]):
            streak[i] = 0
            continue
        if returns.iloc[i] > 0:
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
    percent_rank = pd.Series(returns).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) if len(x) > 1 else 0.5,
        raw=False
    ).values * 100.0
    
    crsi = (rsi_close + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for primary trend direction
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    adx_14 = calculate_adx(high, low, close, 14)
    hma_4h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(hma_4h[i]) or np.isnan(adx_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 12H TREND BIAS (HMA slope over 3 bars) ===
        hma_12h_slope_bull = hma_12h_aligned[i] > hma_12h_aligned[i-3] if i >= 3 else False
        hma_12h_slope_bear = hma_12h_aligned[i] < hma_12h_aligned[i-3] if i >= 3 else False
        
        # Price relative to 12h HMA
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === CHOPPINESS REGIME (single threshold) ===
        is_range = chop_14[i] > 50.0  # Range/consolidation
        is_trend = chop_14[i] <= 50.0  # Trending
        
        # === ADX TREND STRENGTH ===
        is_strong_trend = adx_14[i] > 25.0
        is_weak_trend = adx_14[i] <= 20.0
        
        # === CONNORS RSI (relaxed thresholds) ===
        crsi_oversold = crsi[i] < 20.0  # Oversold
        crsi_overbought = crsi[i] > 80.0  # Overbought
        crsi_neutral_low = crsi[i] < 35.0  # Pullback long
        crsi_neutral_high = crsi[i] > 65.0  # Pullback short
        
        # === 4H HMA SLOPE (2 bars for responsiveness) ===
        hma_4h_slope_bull = hma_4h[i] > hma_4h[i-2] if i >= 2 else False
        hma_4h_slope_bear = hma_4h[i] < hma_4h[i-2] if i >= 2 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Range market (CHOP > 50) + CRSI oversold (< 20) = mean revert long
        if is_range and crsi_oversold:
            new_signal = POSITION_SIZE
        
        # Regime 2: Trending market (CHOP <= 50) + ADX strong + 12h bull + pullback
        elif is_trend and is_strong_trend and hma_12h_slope_bull and price_above_hma_12h:
            if hma_4h_slope_bull and crsi_neutral_low:
                new_signal = POSITION_SIZE
        
        # Regime 3: Weak trend but 12h bull + CRSI very oversold = contrarian long
        elif is_weak_trend and hma_12h_slope_bull and crsi_oversold:
            new_signal = POSITION_SIZE * 0.5  # Half size for weaker signal
        
        # --- SHORT ENTRY ---
        # Regime 1: Range market (CHOP > 50) + CRSI overbought (> 80) = mean revert short
        elif is_range and crsi_overbought:
            new_signal = -POSITION_SIZE
        
        # Regime 2: Trending market (CHOP <= 50) + ADX strong + 12h bear + pullback
        elif is_trend and is_strong_trend and hma_12h_slope_bear and price_below_hma_12h:
            if hma_4h_slope_bear and crsi_neutral_high:
                new_signal = -POSITION_SIZE
        
        # Regime 3: Weak trend but 12h bear + CRSI very overbought = contrarian short
        elif is_weak_trend and hma_12h_slope_bear and crsi_overbought:
            new_signal = -POSITION_SIZE * 0.5  # Half size for weaker signal
        
        # === HOLD POSITION LOGIC ===
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
            if hma_12h_slope_bear and price_below_hma_12h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_12h_slope_bull and price_above_hma_12h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            else:
                # Same side, update extremes
                if position_side > 0:
                    highest_since_entry = max(highest_since_entry, close[i])
                else:
                    lowest_since_entry = min(lowest_since_entry, close[i]) if lowest_since_entry > 0 else close[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals