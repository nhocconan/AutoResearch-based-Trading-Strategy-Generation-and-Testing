#!/usr/bin/env python3
"""
Experiment #671: 4h Primary + 1d HTF — Simplified CRSI Mean Reversion + Choppiness Regime

Hypothesis: After 587 failed strategies, the pattern is clear:
1. #669 (mtf_4h_chop_crsi_hma_1d_v2) got Sharpe=0.151 — CRSI+Chop works on 4h
2. Too many filters = 0 trades (#665, #670 both Sharpe=0.000)
3. Current best Sharpe=0.520 uses 1d CRSI+Chop — need to adapt for 4h with simpler logic

This strategy SIMPLIFIES entry conditions to ensure trades are generated:
- CRSI extremes (<12 or >88) are primary trigger (proven 75% win rate)
- Choppiness used for regime CONTEXT not hard filter (range=trend-follow bias, trend=mean-revert bias)
- 1d HMA for major trend bias only (no slope requirement — too restrictive)
- Asymmetric position sizing based on regime confidence
- Fixed stoploss tracking (bug in previous version)

Why this might beat Sharpe=0.520:
- Simpler entry = more trades (target 30-50/year on 4h)
- CRSI mean-reversion works in both bull and bear markets
- 1d HMA keeps us on right side of major trends without over-filtering
- Conservative sizing (0.25-0.30) controls drawdown through 2022 crash

Position sizing: 0.25 base, 0.30 high conviction (per Rule 4, max 0.40)
Target: 30-50 trades/year on 4h
Stoploss: 2.5*ATR from entry price
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_hma_1d_simp_v1"
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
    def percent_rank(x):
        if len(x) < 2:
            return 0.5
        return (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1)
    
    percent_rank_vals = pd.Series(returns).rolling(window=rank_period, min_periods=rank_period).apply(
        percent_rank, raw=False
    ).values * 100.0
    
    # CRSI
    crsi = (rsi_close + rsi_streak.values + percent_rank_vals) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for primary trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    HIGH_CONV_SIZE = 0.30
    
    # Track position state for stoploss
    entry_price = 0.0
    position_side = 0
    stop_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if atr_14[i] == 0 or atr_14[i] < 1e-10:
            continue
        
        # === 1D TREND BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range = chop_14[i] > 55.0
        is_trend = chop_14[i] < 45.0
        
        # === CONNORS RSI EXTREMES (simplified for more trades) ===
        crsi_oversold = crsi[i] < 12.0
        crsi_overbought = crsi[i] > 88.0
        crsi_moderate_oversold = crsi[i] < 25.0
        crsi_moderate_overbought = crsi[i] > 75.0
        
        # === ENTRY LOGIC (simplified to ensure trades) ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Primary: CRSI extreme oversold in any regime
        if crsi_oversold:
            # Stronger signal if range market or price above 1d HMA
            if is_range or price_above_hma_1d:
                new_signal = HIGH_CONV_SIZE
            else:
                new_signal = BASE_SIZE
        
        # Secondary: Moderate oversold + bullish context
        elif crsi_moderate_oversold and price_above_hma_1d and is_range:
            new_signal = BASE_SIZE
        
        # --- SHORT ENTRY ---
        # Primary: CRSI extreme overbought in any regime
        elif crsi_overbought:
            # Stronger signal if range market or price below 1d HMA
            if is_range or price_below_hma_1d:
                new_signal = -HIGH_CONV_SIZE
            else:
                new_signal = -BASE_SIZE
        
        # Secondary: Moderate overbought + bearish context
        elif crsi_moderate_overbought and price_below_hma_1d and is_range:
            new_signal = -BASE_SIZE
        
        # === HOLD POSITION LOGIC ===
        if position_side != 0 and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR from entry) ===
        stoploss_triggered = False
        
        if position_side > 0 and entry_price > 0:
            stop_price = entry_price - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if position_side < 0 and entry_price > 0:
            stop_price = entry_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON MAJOR TREND FLIP ===
        if position_side > 0 and price_below_hma_1d and chop_14[i] < 40.0:
            # Strong trending bearish — exit long
            new_signal = 0.0
        
        if position_side < 0 and price_above_hma_1d and chop_14[i] < 40.0:
            # Strong trending bullish — exit short
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if position_side == 0:
                # New position
                entry_price = close[i]
                position_side = np.sign(new_signal)
            elif np.sign(new_signal) != position_side:
                # Position flip
                entry_price = close[i]
                position_side = np.sign(new_signal)
        else:
            # Exit position
            if position_side != 0:
                entry_price = 0.0
                position_side = 0
                stop_price = 0.0
        
        signals[i] = new_signal
    
    return signals