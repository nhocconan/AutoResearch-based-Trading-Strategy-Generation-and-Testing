#!/usr/bin/env python3
"""
Experiment #362: 12h Primary + 1d/1w HTF — Connors RSI Mean Reversion with Dual HMA Bias

Hypothesis: 12h timeframe is underutilized but proven to work (lower fee drag, fewer whipsaws).
Previous 12h strategies failed because:
1. Too many confluence filters prevented trades (0 trades = auto-reject)
2. Choppiness thresholds too complex for 12h (regime switches too slow)
3. Didn't use BOTH 1d AND 1w for macro bias (single HTF insufficient)

This strategy uses Connors RSI (CRSI) for mean reversion with DUAL HTF bias:
1. 1w HMA(21) = ULTRA MACRO BIAS (bull/bear market filter)
2. 1d HMA(21) = MACRO BIAS (medium-term trend)
3. CRSI(3,2,100) for entry timing (extremes <15 long, >85 short)
4. Choppiness Index for regime (CHOP>55=range, CHOP<45=trend) - WIDER thresholds
5. ATR(14) trailing stop at 2.5x for risk management
6. RELAXED entry conditions to ensure 25-50 trades/year on 12h

KEY INSIGHT: CRSI is more reliable than RSI for mean reversion (75% win rate in literature).
Combined with DUAL HTF bias (1d + 1w), this filters out counter-trend trades that fail.
12h timeframe = ~730 bars/year, target 30-50 trades = 4-7% trade rate.

TARGET: 30-50 trades/year on 12h, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_dual_hma_1d1w_chop_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Percentile rank of price change over lookback
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down days)
    delta = close_s.diff()
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_s = pd.Series(streak)
    streak_gain = streak_s.clip(lower=0)
    streak_loss = (-streak_s).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # Component 3: PercentRank of price change
    pct_change = close_s.pct_change()
    percent_rank = pct_change.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).sum() / len(x.iloc[:-1]) if len(x) > 1 else 0.5,
        raw=False
    ) * 100.0
    percent_rank = percent_rank.fillna(50.0).values
    
    # Combine into CRSI
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Calculate and align HTF HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 12h (target 30-50 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (DUAL HTF HMA - HARD FILTER) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index - WIDER thresholds for 12h) ===
        is_choppy = chop[i] > 55.0  # High choppiness = range regime (mean revert)
        is_trending = chop[i] < 45.0  # Low choppiness = trend regime
        is_neutral = not is_choppy and not is_trending  # 45-55 = transition
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # CRSI extremes for mean reversion
        crsi_oversold = crsi[i] < 15.0  # Very oversold
        crsi_overbought = crsi[i] > 85.0  # Very overbought
        
        # === LONG ENTRY CONDITIONS (RELAXED for trade frequency) ===
        long_bias = price_above_hma_1d  # Primary bias: price above 1d HMA
        long_strong_bias = long_bias and price_above_hma_1w  # Both 1d and 1w bullish
        
        if long_strong_bias:
            # Strong bullish: enter on CRSI oversold in any regime
            if crsi_oversold:
                desired_signal = BASE_SIZE
        elif long_bias:
            # Moderate bullish: only enter in choppy regime (mean revert)
            if is_choppy and crsi_oversold:
                desired_signal = BASE_SIZE * 0.8
        
        # === SHORT ENTRY CONDITIONS (RELAXED for trade frequency) ===
        short_bias = price_below_hma_1d  # Primary bias: price below 1d HMA
        short_strong_bias = short_bias and price_below_hma_1w  # Both 1d and 1w bearish
        
        if short_strong_bias:
            # Strong bearish: enter on CRSI overbought in any regime
            if crsi_overbought:
                desired_signal = -BASE_SIZE
        elif short_bias:
            # Moderate bearish: only enter in choppy regime (mean revert)
            if is_choppy and crsi_overbought:
                desired_signal = -BASE_SIZE * 0.8
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === CRSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and crsi[i] > 70:
            # Long position: exit when CRSI reaches overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 30:
            # Short position: exit when CRSI reaches oversold
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if bias still valid
            if position_side > 0:
                if price_above_hma_1d:
                    # Hold long if CRSI not yet overbought
                    if crsi[i] < 70:
                        desired_signal = BASE_SIZE
            elif position_side < 0:
                if price_below_hma_1d:
                    # Hold short if CRSI not yet oversold
                    if crsi[i] > 30:
                        desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals