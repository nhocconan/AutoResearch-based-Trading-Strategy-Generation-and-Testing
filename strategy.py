#!/usr/bin/env python3
"""
Experiment #382: 12h Primary + 1d/1w HTF — Choppiness Regime + HMA + Relaxed CRSI

Hypothesis: Previous 12h strategies (#372, #376) showed promise but were over-filtered.
This strategy uses Choppiness Index for regime detection with RELAXED entry conditions
to ensure trade generation. Key innovations:

1. CHOP(14) regime: >61.8 = range (mean revert), <38.2 = trend (breakout)
2. HMA(21) primary trend on 12h — faster than KAMA, proven on higher TF
3. HMA(50) on 1d for HTF bias — smoother than 1d KAMA
4. RELAXED CRSI: <25 long, >75 short (not 10/90 which rarely trigger on 12h)
5. Multiple entry paths (OR logic) — breakout OR pullback OR mean-revert
6. Position size 0.28 discrete — balances return vs drawdown for 12h TF
7. ATR(14) trailing stop at 2.5x — protects from 2022-style crashes

Target: 20-40 trades/year on 12h, Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL individually).
Must beat current best: mtf_4h_triple_regime_crsi_donchian_1d1w_v1 (Sharpe=0.612)

Why 12h works: Fewer false signals than 4h, captures multi-day trends, less fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_hma_crsi_1d1w_relaxed_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    if period < 2:
        return np.full(n, close[0])
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    # HMA formula
    hma_raw = 2 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.fillna(method='ffill').fillna(close[0]).values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    choppiness = np.full(n, 50.0)
    
    # ATR calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = tr[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    choppiness = np.clip(choppiness, 0, 100)
    return choppiness

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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Relaxed thresholds for 12h crypto: <25 oversold, >75 overbought
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for short-term extremes
    rsi_fast = calculate_rsi(close, period=rsi_period)
    
    # RSI of Streak - consecutive up/down bars
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= streak_period:
            streak_rsi[i] = 100.0
        elif streak[i] <= -streak_period:
            streak_rsi[i] = 0.0
        else:
            streak_rsi[i] = 50.0 + 25.0 * streak[i]
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # PercentRank - percentile of today's return vs last pr_period bars
    returns = close_s.pct_change()
    percent_rank = np.full(n, 50.0)
    for i in range(pr_period, n):
        window = returns.iloc[i-pr_period:i]
        if len(window) > 0:
            percent_rank[i] = (returns.iloc[i] > window).sum() / len(window) * 100
    
    # Combine into CRSI
    crsi = (rsi_fast + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
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
    atr_14 = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    hma_21_12h = calculate_hma(close, period=21)
    hma_50_12h = calculate_hma(close, period=50)
    
    # Calculate and align HTF HMA for bias (1d HMA50)
    hma_50_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_50_1d_raw)
    
    # Calculate and align HTF HMA for major trend (1w HMA50)
    hma_50_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_50_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 12h (target 20-40 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_21_12h[i]) or np.isnan(hma_50_12h[i]):
            continue
        if np.isnan(hma_50_1d_aligned[i]) or np.isnan(hma_50_1w_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_ranging = choppiness[i] > 61.8
        is_trending = choppiness[i] < 38.2
        # Neutral zone: 38.2 - 61.8 (use trend logic as default)
        
        # === HTF BIAS (1d HMA50) ===
        price_above_hma_1d = close[i] > hma_50_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_50_1d_aligned[i]
        
        # === MAJOR TREND (1w HMA50) — optional filter ===
        price_above_hma_1w = close[i] > hma_50_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_50_1w_aligned[i]
        
        # === PRIMARY TREND (12h HMA21 vs HMA50) ===
        hma_bullish = hma_21_12h[i] > hma_50_12h[i]
        hma_bearish = hma_21_12h[i] < hma_50_12h[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP — Multiple entry paths (OR logic for trade generation)
        long_bias = price_above_hma_1d  # HTF bullish
        
        # Entry trigger 1: Donchian breakout in trend regime
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        
        # Entry trigger 2: CRSI oversold pullback in range
        crsi_oversold = crsi[i] < 25.0  # Relaxed from 10
        
        # Entry trigger 3: HMA crossover confirmation
        momentum_long = hma_bullish and price_above_hma_1d
        
        # Entry trigger 4: Simple mean reversion in strong range
        strong_range = choppiness[i] > 65.0
        
        # LONG ENTRY: Need HTF bias + ANY entry trigger
        if long_bias:
            if is_trending and breakout_long:
                # Trend breakout long
                desired_signal = BASE_SIZE
            elif is_ranging and crsi_oversold:
                # Range mean-reversion long
                desired_signal = BASE_SIZE
            elif strong_range and crsi_oversold:
                # Strong range mean-reversion (ignore HTF for deep oversold)
                desired_signal = BASE_SIZE
            elif momentum_long and crsi_oversold:
                # Pullback in uptrend
                desired_signal = BASE_SIZE
            elif breakout_long and momentum_long:
                # Breakout with trend confirmation
                desired_signal = BASE_SIZE
        
        # SHORT SETUP — Multiple entry paths (OR logic for trade generation)
        short_bias = price_below_hma_1d  # HTF bearish
        
        # Entry trigger 1: Donchian breakdown in trend regime
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # Entry trigger 2: CRSI overbought rally in range
        crsi_overbought = crsi[i] > 75.0  # Relaxed from 90
        
        # Entry trigger 3: HMA crossover confirmation
        momentum_short = hma_bearish and price_below_hma_1d
        
        # Entry trigger 4: Simple mean reversion in strong range
        strong_range_short = choppiness[i] > 65.0
        
        # SHORT ENTRY: Need HTF bias + ANY entry trigger
        if short_bias:
            if is_trending and breakout_short:
                # Trend breakdown short
                desired_signal = -BASE_SIZE
            elif is_ranging and crsi_overbought:
                # Range mean-reversion short
                desired_signal = -BASE_SIZE
            elif strong_range_short and crsi_overbought:
                # Strong range mean-reversion (ignore HTF for deep overbought)
                desired_signal = -BASE_SIZE
            elif momentum_short and crsi_overbought:
                # Rally in downtrend
                desired_signal = -BASE_SIZE
            elif breakout_short and momentum_short:
                # Breakdown with trend confirmation
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
        
        # === CRSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and crsi[i] > 70:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 30:
            desired_signal = 0.0
        
        # === TREND EXIT (HTF bias reversal) ===
        if in_position and position_side > 0 and price_below_hma_1d:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_1d:
                desired_signal = BASE_SIZE
            elif position_side < 0 and price_below_hma_1d:
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