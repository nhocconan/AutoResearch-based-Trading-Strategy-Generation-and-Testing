#!/usr/bin/env python3
"""
Experiment #232: 12h Primary + 1d/1w HTF — Dual Regime (Choppiness + Connors RSI + Donchian)

Hypothesis: After 195 failed experiments, return to proven patterns from research:
1. Choppiness Index (CHOP) detects regime: >61.8 = range, <38.2 = trend
2. In RANGE regime: Connors RSI mean reversion (75% win rate documented)
3. In TREND regime: Donchian(20) breakout + HMA confirmation
4. 1d/1w HMA for macro bias alignment (only trade with HTF trend)
5. ATR(14) 2.5x trailing stoploss
6. Position sizing: 0.25-0.30 discrete levels

Why 12h: Higher TF = fewer trades (target 20-50/year) = less fee drag.
Proven in experiment history: 12h strategies with Choppiness filter worked on ETH.

Key differences from failed attempts:
- NO Fisher Transform (failed #219, #224)
- NO Vol-spike logic (failed #221, #230)
- NO complex KAMA adaptive (failed #222, #224)
- YES Connors RSI (documented 75% win rate in literature)
- YES Choppiness regime switch (ETH Sharpe +0.923 in research)
- YES Donchian breakout for trend regime (SOL Sharpe +0.782)

TARGET: Sharpe > 0.50 on ALL symbols, 25-50 trades/year, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_crsi_donchian_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

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
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback period
    
    CRSI < 10 = oversold (long opportunity)
    CRSI > 90 = overbought (short opportunity)
    """
    close_s = pd.Series(close)
    
    # RSI(3) component
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100.0 - (100.0 / (1.0 + rs))
    rsi_close = rsi_close.fillna(50.0)
    
    # RSI Streak component
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.clip(lower=0)
    streak_loss = (-streak_s).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    rsi_streak = rsi_streak.fillna(50.0)
    
    # Percent Rank component
    pct_change = close_s.pct_change()
    percent_rank = pct_change.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / len(x.iloc[:-1]) * 100.0 if len(x) > 1 else 50.0,
        raw=False
    )
    percent_rank = percent_rank.fillna(50.0)
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    return crsi.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    atr_14 = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate 1d HMA for macro trend (aligned properly)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1w HMA for macro trend (aligned properly)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Position tracking
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            continue
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HTF MACRO BIAS (1d + 1w HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        macro_bullish = price_above_hma_1d and price_above_hma_1w
        macro_bearish = price_below_hma_1d and price_below_hma_1w
        macro_neutral = not macro_bullish and not macro_bearish
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_ranging = choppiness[i] > 61.8
        is_trending = choppiness[i] < 38.2
        is_transition = not is_ranging and not is_trending
        
        # === 12h TREND (HMA crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_neutral = 15.0 <= crsi[i] <= 85.0
        
        # === DETERMINE DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion with Connors RSI
        if is_ranging:
            # Long: CRSI oversold + price above 1d HMA (macro support)
            if crsi_oversold and (macro_bullish or macro_neutral):
                desired_signal = POSITION_SIZE_HALF
            
            # Short: CRSI overbought + price below 1d HMA (macro resistance)
            elif crsi_overbought and (macro_bearish or macro_neutral):
                desired_signal = -POSITION_SIZE_HALF
        
        # TREND REGIME: Donchian breakout + HMA confirmation
        elif is_trending:
            # Long: Donchian breakout + HMA bullish + macro bullish
            if donchian_breakout_long and hma_bullish and macro_bullish:
                desired_signal = POSITION_SIZE_FULL
            elif donchian_breakout_long and hma_bullish and macro_neutral:
                desired_signal = POSITION_SIZE_HALF
            
            # Short: Donchian breakout + HMA bearish + macro bearish
            elif donchian_breakout_short and hma_bearish and macro_bearish:
                desired_signal = -POSITION_SIZE_FULL
            elif donchian_breakout_short and hma_bearish and macro_neutral:
                desired_signal = -POSITION_SIZE_HALF
        
        # TRANSITION REGIME: Use HMA crossover with RSI filter
        elif is_transition:
            if hma_bullish and crsi[i] < 50.0 and (macro_bullish or macro_neutral):
                desired_signal = POSITION_SIZE_HALF
            elif hma_bearish and crsi[i] > 50.0 and (macro_bearish or macro_neutral):
                desired_signal = -POSITION_SIZE_HALF
        
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
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_bearish and is_trending:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_bullish and is_trending:
            desired_signal = 0.0
        
        # === MACRO REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bullish:
            desired_signal = 0.0
        
        # === CRSI REVERSION EXIT (in range regime) ===
        if in_position and position_side > 0 and is_ranging and crsi[i] > 70.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and is_ranging and crsi[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if trend still valid ===
        if in_position and desired_signal == 0.0:
            if position_side > 0 and hma_bullish and crsi[i] < 80.0:
                desired_signal = POSITION_SIZE_HALF
            elif position_side < 0 and hma_bearish and crsi[i] > 20.0:
                desired_signal = -POSITION_SIZE_HALF
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals