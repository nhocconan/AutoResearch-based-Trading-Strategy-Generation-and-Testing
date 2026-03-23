#!/usr/bin/env python3
"""
Experiment #343: 1d Primary + 1w HTF — Connors RSI Mean Reversion with Regime Filter

Hypothesis: Previous 4h/12h strategies failed because:
1. Too many conflicting regime filters reduced trade count below viable levels
2. Simple trend following gets whipsawed in 2022 crash and 2025 bear market
3. CRSI mean reversion showed +0.923 Sharpe on ETH in research but wasn't properly implemented

This strategy uses:
1. 1w HMA(21) as MACRO BIAS (hard filter: only long if 1w bullish, only short if 1w bearish)
2. Connors RSI (CRSI) for entry timing: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3. Choppiness Index for regime detection: CHOP>61.8=range (mean revert), CHOP<38.2=trend
4. 1d ATR(14) for stoploss (2.5x ATR trailing)
5. Asymmetric logic: in bear macro (price<1w HMA), only short rallies; in bull macro, only long dips

KEY INSIGHT: Connors RSI has 75% win rate on mean reversion. Combined with 1w trend bias,
this captures counter-trend moves within the larger trend direction. Works in both bull and bear.

TARGET: 20-40 trades/year on 1d, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_regime_1w_hma_chop_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    hull = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hull.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

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
    PercentRank: percentile rank of today's change vs last 100 days
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI
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
        up_streaks = np.sum((streak[max(0,i-streak_period):i+1] > 0).astype(float))
        total_streaks = streak_period
        if total_streaks > 0:
            streak_rsi[i] = 100.0 * up_streaks / total_streaks
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank
    pct_rank = np.zeros(n)
    for i in range(rank_period, n):
        today_change = close[i] - close[i-1]
        past_changes = close[i-rank_period:i] - close[i-rank_period-1:i-1]
        pct_rank[i] = 100.0 * np.sum(past_changes < today_change) / rank_period
    
    # Combine
    crsi = (rsi_short + streak_rsi + pct_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Calculate and align 1w HMA for macro bias (HARD FILTER)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 1d
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need enough data for CRSI rank_period=100
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1w HMA - HARD FILTER) ===
        # Only take LONGS if price above 1w HMA (bullish macro)
        # Only take SHORTS if price below 1w HMA (bearish macro)
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 61.8  # High choppiness = range/mean revert
        is_trending = chop[i] < 38.2  # Low choppiness = trend
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 10 = extremely oversold (long signal)
        # CRSI > 90 = extremely overbought (short signal)
        # CRSI < 30 = oversold (weak long)
        # CRSI > 70 = overbought (weak short)
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        crsi_mild_oversold = crsi[i] < 30
        crsi_mild_overbought = crsi[i] > 70
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: Mean reversion at CRSI extremes
            # Only take trades aligned with 1w bias
            
            if price_above_hma_1w and crsi_oversold:
                # Long extremely oversold in bullish macro
                desired_signal = BASE_SIZE
            
            elif price_below_hma_1w and crsi_overbought:
                # Short extremely overbought in bearish macro
                desired_signal = -BASE_SIZE
            
            # Smaller positions for mild extremes
            elif price_above_hma_1w and crsi_mild_oversold and desired_signal == 0:
                desired_signal = BASE_SIZE * 0.5
            
            elif price_below_hma_1w and crsi_mild_overbought and desired_signal == 0:
                desired_signal = -BASE_SIZE * 0.5
        
        else:
            # TRENDING REGIME: Wait for pullbacks in direction of 1w trend
            
            if price_above_hma_1w and crsi_mild_oversold:
                # Long pullback in bullish macro
                desired_signal = BASE_SIZE
            
            elif price_below_hma_1w and crsi_mild_overbought:
                # Short rally in bearish macro
                desired_signal = -BASE_SIZE
        
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
        
        # === CRSI REVERSAL EXIT ===
        # Exit long when CRSI goes overbought
        # Exit short when CRSI goes oversold
        if in_position and position_side > 0 and crsi[i] > 80:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if bias still valid
            if position_side > 0 and price_above_hma_1w:
                desired_signal = BASE_SIZE * 0.5  # Reduce size on hold
            elif position_side < 0 and price_below_hma_1w:
                desired_signal = -BASE_SIZE * 0.5
        
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