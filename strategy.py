#!/usr/bin/env python3
"""
Experiment #001: 4h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime Switch

Hypothesis: Combining Connors RSI (proven 75% win rate mean reversion) with 
Choppiness Index regime detection will outperform pure vol-spike strategies.
CRSI captures oversold/overbought extremes better than standard RSI.
Choppiness Index switches between mean-revert (CHOP>61.8) and trend-follow (CHOP<38.2).

Key improvements over previous attempts:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — research-backed
2. Choppiness Index regime filter — avoids mean-revert in strong trends
3. 1d HMA for primary trend bias (simpler than 12h+1d dual)
4. Cleaner entry logic: CRSI<15 long, CRSI>85 short, with regime confirmation
5. Position size 0.30 (discrete, within 0.20-0.35 range per Rule 4)

Why this might work:
- CRSI mean reversion worked through 2022 crash (unlike pure trend)
- Choppiness filter prevents entering mean-revert in strong trends
- 4h TF targets 20-50 trades/year (fee-efficient per Rule 10)
- Simpler logic = more reliable trade generation (Rule 9)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_1d_v1"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Research shows CRSI<10 and CRSI>90 are extreme mean-reversion signals
    with 70-75% win rate on BTC/ETH.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] > 0:
            # Up streak - calculate RSI of streak length
            lookback = streak[i]
            streak_rsi[i] = min(100, 50 + lookback * 10)
        elif streak[i] < 0:
            lookback = -streak[i]
            streak_rsi[i] = max(0, 50 - lookback * 10)
        else:
            streak_rsi[i] = 50
    
    # Percent Rank - where current close ranks in last 100 bars
    pct_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        pct_rank[i] = 100.0 * np.sum(window[:-1] < close[i]) / (rank_period - 1)
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    atr_vals = calculate_atr(high, low, close, period=period)
    
    chop = np.zeros(len(close))
    for i in range(period, len(close)):
        atr_sum = np.sum(atr_vals[i-period+1:i+1])
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        range_hl = highest - lowest
        
        if range_hl > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    pct_b = (close - lower) / (upper - lower + 1e-10)
    return upper.values, lower.values, pct_b.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for primary trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for macro regime
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_lower, bb_pct_b = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    rsi_14 = calculate_rsi(close, period=14)
    
    # 4h HMA for local trend
    hma_4h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(atr_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # 1d HMA slope
        hma_1d_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 else False
        hma_1d_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 else False
        
        # === MACRO REGIME (1w HMA) ===
        macro_bullish = hma_1w_aligned[i] > hma_1w_aligned[i-2] if i >= 2 else False
        macro_bearish = hma_1w_aligned[i] < hma_1w_aligned[i-2] if i >= 2 else False
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0  # Range market (mean reversion favored)
        is_trending = chop[i] < 45.0  # Trending market (trend follow favored)
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 20  # Extreme oversold
        crsi_overbought = crsi[i] > 80  # Extreme overbought
        
        # === BOLLINGER BAND CONFIRMATION ===
        bb_extreme_low = bb_pct_b[i] < 0.10
        bb_extreme_high = bb_pct_b[i] > 0.90
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Mean reversion in choppy market
        long_mr = is_choppy and crsi_oversold and bb_extreme_low
        
        # Trend pullback in trending market
        long_trend = is_trending and crsi_oversold and price_above_hma_1d and hma_1d_bull
        
        # Macro bullish override (stronger signal)
        long_macro = macro_bullish and crsi_oversold and rsi_14[i] < 35
        
        if long_mr or long_trend or long_macro:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Mean reversion in choppy market
        short_mr = is_choppy and crsi_overbought and bb_extreme_high
        
        # Trend pullback in trending market
        short_trend = is_trending and crsi_overbought and price_below_hma_1d and hma_1d_bear
        
        # Macro bearish override (stronger signal)
        short_macro = macro_bearish and crsi_overbought and rsi_14[i] > 65
        
        if short_mr or short_trend or short_macro:
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION (avoid churning) ===
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
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_1d_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_bull and price_above_hma_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
        else:
            if in_position and prev_signal != 0.0:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals