#!/usr/bin/env python3
"""
Experiment #132: 12h Primary + 1d/1w HTF — KAMA Trend + Choppiness + Connors RSI

Hypothesis: Previous complex regime strategies failed due to over-filtering (0 trades).
This combines proven 12h patterns with simpler logic:

1) 1d HMA(21) for macro trend bias — only trade in trend direction
2) 12h KAMA(14) for adaptive trend following — adapts to volatility automatically
3) Choppiness Index(14) — CHOP>61.8 = range (mean revert), CHOP<38.2 = trend
4) Connors RSI for entry timing — (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long: CRSI<15, Short: CRSI>85 (proven 75% win rate in research)
5) ATR(14) trailing stop at 2.5x — locks profits, limits drawdown

Why 12h works:
- Natural 25-50 trades/year (low fee drag at 0.05% RT)
- Less noise than 4h/1h, more signals than 1d
- KAMA adapts to crypto volatility better than EMA/HMA
- CRSI catches reversals in bear/range markets (2025 test period)

Position size: 0.25 base, 0.30 max with strong confluence
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_chop_crsi_1d1w_v1"
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

def calculate_kama(close, period=14, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.nan
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=period, min_periods=period).sum().values
    er = change / (volatility + 1e-10)
    er[0:period] = 0.0
    
    # Smoothing constant
    sc = (er * (2.0/(fast+1) - 2.0/(slow+1)) + 2.0/(slow+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (100 - 100 * sum(ATR) / (max_high - min_low))."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    chop = 100.0 - 100.0 * (atr_sum / (highest - lowest + 1e-10))
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3."""
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (consecutive up/down days)
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
    for i in range(rank_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = 100.0 * streak_abs[i] / (rank_period + 1e-10)
        else:
            streak_rsi[i] = 100.0 - (100.0 * streak_abs[i] / (rank_period + 1e-10))
    streak_rsi = np.clip(streak_rsi, 0, 100)
    streak_rsi_s = pd.Series(streak_rsi).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    # Percent Rank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        percent_rank[i] = rank
    
    # Connors RSI
    crsi = (rsi_3.values + streak_rsi_s + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for weekly trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_14 = calculate_kama(close, period=14, fast=2, slow=30)
    kama_50 = calculate_kama(close, period=50, fast=2, slow=30)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_14[i]) or np.isnan(kama_50[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === HTF TREND BIAS (1d & 1w HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 12h TREND FILTER (KAMA) ===
        kama_bullish = kama_14[i] > kama_50[i]
        kama_bearish = kama_14[i] < kama_50[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 61.8  # Range market
        is_trending = chop_14[i] < 38.2  # Trend market
        is_neutral = not is_choppy and not is_trending
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Long entry
        crsi_overbought = crsi[i] > 85.0  # Short entry
        crsi_neutral = 15.0 <= crsi[i] <= 85.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Trending regime: KAMA bullish + 1d/1w trend up + CRSI pullback
        if is_trending:
            if kama_bullish and price_above_hma_1d and crsi_oversold:
                new_signal = POSITION_SIZE_BASE
                if price_above_hma_1w:
                    new_signal = POSITION_SIZE_MAX
        
        # Range regime: Mean reversion at CRSI extreme + 1d trend support
        elif is_choppy:
            if crsi_oversold and price_above_hma_1d:
                new_signal = POSITION_SIZE_BASE
        
        # Neutral regime: Require stronger confluence
        elif is_neutral:
            if kama_bullish and price_above_hma_1d and price_above_hma_1w and crsi_oversold:
                new_signal = POSITION_SIZE_BASE
        
        # --- SHORT ENTRY ---
        # Trending regime: KAMA bearish + 1d/1w trend down + CRSI rally
        if is_trending:
            if kama_bearish and price_below_hma_1d and crsi_overbought:
                new_signal = -POSITION_SIZE_BASE
                if price_below_hma_1w:
                    new_signal = -POSITION_SIZE_MAX
        
        # Range regime: Mean reversion at CRSI extreme + 1d trend resistance
        elif is_choppy:
            if crsi_overbought and price_below_hma_1d:
                new_signal = -POSITION_SIZE_BASE
        
        # Neutral regime: Require stronger confluence
        elif is_neutral:
            if kama_bearish and price_below_hma_1d and price_below_hma_1w and crsi_overbought:
                new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold long if KAMA still bullish and CRSI not overbought
        if in_position and new_signal == 0.0:
            if position_side > 0:
                if kama_bullish and crsi[i] < 85.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                if kama_bearish and crsi[i] > 15.0:
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
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            if kama_bearish and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if kama_bullish and price_above_hma_1d:
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