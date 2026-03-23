#!/usr/bin/env python3
"""
Experiment #272: 12h Primary + 1d/1w HTF — Connors RSI Mean Reversion with Trend Filter

Hypothesis: Previous HMA crossover strategies failed because they're trend-following in 
range/bear markets (2022 crash, 2025 bear). Connors RSI (CRSI) is a proven mean-reversion 
indicator with 75% win rate in research literature. Combined with 1d/1w HMA trend filters,
this should capture pullbacks in established trends while avoiding counter-trend trades.

KEY DIFFERENCES from failed experiments:
- CRSI instead of regular RSI (CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3)
- 12h primary timeframe (lower trade frequency = less fee drag, 20-50 trades/year target)
- 1d HMA(21) for intermediate trend bias (not hard filter, soft weight)
- 1w HMA(50) for macro regime detection
- ATR(14) 2.5x trailing stoploss
- Position size: 0.28 (conservative for 12h volatility)

ENTRY LOGIC:
- Long: CRSI < 15 (oversold) + price > 1d HMA(21) (uptrend pullback)
- Short: CRSI > 85 (overbought) + price < 1d HMA(21) (downtrend rally)
- 1w HMA(50) confirms macro direction (soft filter)

TARGET: Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL), 25-50 trades/year, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_hma_regime_1d1w_atr_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - fast RSI for mean reversion
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI Streak - measures consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI of streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # Percent Rank - today's return vs last 100 days
    returns = close_s.pct_change()
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current_return = returns.iloc[i]
        if np.isnan(current_return):
            percent_rank[i] = 50.0
        else:
            rank = (window < current_return).sum()
            percent_rank[i] = (rank / rank_period) * 100.0
    
    # CRSI = average of three components
    crsi = (rsi_fast + rsi_streak + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    
    return crsi

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
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Calculate and align 1d HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, 50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Conservative for 12h volatility
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi_12h[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO REGIME (1w HMA) - SOFT FILTER ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA) - PRIMARY FILTER ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CRSI MEAN REVERSION SIGNALS ===
        # CRSI < 15 = extremely oversold (long opportunity in uptrend)
        # CRSI > 85 = extremely overbought (short opportunity in downtrend)
        crsi_oversold = crsi_12h[i] < 15.0
        crsi_overbought = crsi_12h[i] > 85.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: CRSI oversold + price above 1d HMA (uptrend pullback)
        # 1w HMA confirms macro bull (soft - can enter without it but prefer with)
        if crsi_oversold and price_above_hma_1d:
            if price_above_hma_1w:
                desired_signal = POSITION_SIZE  # Strong signal with macro confirmation
            else:
                desired_signal = POSITION_SIZE * 0.7  # Weaker signal without macro
        
        # SHORT ENTRY: CRSI overbought + price below 1d HMA (downtrend rally)
        # 1w HMA confirms macro bear (soft - can enter without it but prefer with)
        elif crsi_overbought and price_below_hma_1d:
            if price_below_hma_1w:
                desired_signal = -POSITION_SIZE  # Strong signal with macro confirmation
            else:
                desired_signal = -POSITION_SIZE * 0.7  # Weaker signal without macro
        
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
        
        # === CRSI MEAN REVERSION EXIT (take profit) ===
        # Exit long when CRSI rises above 50 (mean reached)
        # Exit short when CRSI falls below 50 (mean reached)
        if in_position and position_side > 0 and crsi_12h[i] > 50.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi_12h[i] < 50.0:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price falls below 1d HMA (trend broken)
        # Exit short if price rises above 1d HMA (trend broken)
        if in_position and position_side > 0 and price_below_hma_1d:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        # If already in position and no exit trigger, maintain position
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_1d and crsi_12h[i] <= 50.0:
                desired_signal = POSITION_SIZE if price_above_hma_1w else POSITION_SIZE * 0.7
            elif position_side < 0 and price_below_hma_1d and crsi_12h[i] >= 50.0:
                desired_signal = -POSITION_SIZE if price_below_hma_1w else -POSITION_SIZE * 0.7
        
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