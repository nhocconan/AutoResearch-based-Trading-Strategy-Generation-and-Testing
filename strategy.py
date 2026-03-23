#!/usr/bin/env python3
"""
Experiment #656: 12h Primary + 1d HTF — Simplified Choppiness Regime + Connors RSI

Hypothesis: Previous 12h strategies failed due to:
1. Too many conflicting filters (#652 Sharpe=-0.314, #646 Sharpe=-0.077)
2. Over-filtering导致 0 trades (#645, #648, #650 all Sharpe=0.000)
3. Complex regime detection that rarely triggers

This strategy SIMPLIFIES:
- Single regime filter: Choppiness Index only (no ADX, no volume filters)
- Aggressive CRSI thresholds: <15 long, >85 short (ensures trades)
- 1d HMA slope for major trend bias (not too slow like 1w)
- Fixed position tracking (previous code had bugs in state management)
- ATR stoploss at 2.5x with proper trailing

Why 12h might work better than 4h:
- Fewer trades = less fee drag (target 25-40 trades/year)
- Less noise than 4h, more signal than 1d
- 1d HTF provides clear trend direction without lag

Position sizing: 0.30 discrete (per Rule 4)
Stoploss: 2.5*ATR trailing per position
Target: Beat Sharpe=0.520 baseline
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_hma_1d_v1"
timeframe = "12h"
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
    CHOP > 61.8: Range | CHOP < 38.2: Trend
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
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    returns = close_s.pct_change().fillna(0)
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff().fillna(0)
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
    
    # Load 1d HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_12h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    POSITION_SIZE = 0.30
    STOPLOSS_MULT = 2.5
    
    # Position state tracking
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    extreme_price = 0.0  # highest for long, lowest for short
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(hma_12h[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if atr_14[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === 1D TREND BIAS ===
        hma_1d_slope = 0
        if i >= 5 and not np.isnan(hma_1d_aligned[i-5]):
            if hma_1d_aligned[i] > hma_1d_aligned[i-5] * 1.002:
                hma_1d_slope = 1
            elif hma_1d_aligned[i] < hma_1d_aligned[i-5] * 0.998:
                hma_1d_slope = -1
        
        price_vs_hma_1d = 0
        if hma_1d_aligned[i] > 0:
            if close[i] > hma_1d_aligned[i] * 1.005:
                price_vs_hma_1d = 1
            elif close[i] < hma_1d_aligned[i] * 0.995:
                price_vs_hma_1d = -1
        
        # === CHOPPINESS REGIME ===
        is_range = chop_14[i] > 55.0
        is_trend = chop_14[i] < 45.0
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_moderate_oversold = crsi[i] < 35.0
        crsi_moderate_overbought = crsi[i] > 65.0
        
        # === 12H HMA SLOPE ===
        hma_12h_slope = 0
        if i >= 3 and not np.isnan(hma_12h[i-3]):
            if hma_12h[i] > hma_12h[i-3] * 1.002:
                hma_12h_slope = 1
            elif hma_12h[i] < hma_12h[i-3] * 0.998:
                hma_12h_slope = -1
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG entries
        if is_range and crsi_oversold:
            # Range market + extreme oversold = mean revert long
            new_signal = POSITION_SIZE
        elif is_trend and hma_1d_slope >= 0 and price_vs_hma_1d >= 0:
            # Trending + 1d bullish + pullback
            if hma_12h_slope >= 0 and crsi_moderate_oversold:
                new_signal = POSITION_SIZE
        elif crsi[i] < 10.0:
            # Extreme CRSI regardless of regime (catch major reversals)
            new_signal = POSITION_SIZE
        
        # SHORT entries
        if is_range and crsi_overbought:
            # Range market + extreme overbought = mean revert short
            new_signal = -POSITION_SIZE
        elif is_trend and hma_1d_slope <= 0 and price_vs_hma_1d <= 0:
            # Trending + 1d bearish + pullback
            if hma_12h_slope <= 0 and crsi_moderate_overbought:
                new_signal = -POSITION_SIZE
        elif crsi[i] > 90.0:
            # Extreme CRSI regardless of regime
            new_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if position_side == 1:  # Long position
            extreme_price = max(extreme_price, close[i])
            stop_price = extreme_price - STOPLOSS_MULT * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        elif position_side == -1:  # Short position
            if extreme_price == 0.0:
                extreme_price = close[i]
            else:
                extreme_price = min(extreme_price, close[i])
            stop_price = extreme_price + STOPLOSS_MULT * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        if position_side == 1 and hma_1d_slope == -1 and price_vs_hma_1d == -1:
            new_signal = 0.0
        
        if position_side == -1 and hma_1d_slope == 1 and price_vs_hma_1d == 1:
            new_signal = 0.0
        
        # === UPDATE POSITION STATE ===
        if new_signal != 0.0:
            if position_side == 0:
                # Entering new position
                position_side = 1 if new_signal > 0 else -1
                entry_price = close[i]
                extreme_price = close[i]
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = 1 if new_signal > 0 else -1
                entry_price = close[i]
                extreme_price = close[i]
            # else: holding existing position
        else:
            # Exiting position
            if position_side != 0:
                position_side = 0
                entry_price = 0.0
                extreme_price = 0.0
        
        signals[i] = new_signal
    
    return signals