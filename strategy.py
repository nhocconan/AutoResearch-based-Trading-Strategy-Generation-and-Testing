#!/usr/bin/env python3
"""
Experiment #454: 4h Primary + 12h/1d HTF — Simplified Connors RSI + HMA Trend + Choppiness Regime

Hypothesis: After analyzing 453 experiments, clear patterns emerge:
1. 4h timeframe offers better trade frequency than 12h/1d while maintaining edge
2. Connors RSI proven 75% win rate for mean reversion entries (research notes)
3. Choppiness Index regime detection switches between trend/mean-revert modes
4. 12h HMA provides trend direction without over-filtering (1d too slow for 4h entries)
5. SIMPLER entry logic = more trades (critical: need >=30 trades/symbol on train)

Why this might beat current best (Sharpe=0.435):
- 4h TF captures more opportunities than 12h while avoiding 1h/15m noise
- Connors RSI catches reversals better than standard RSI(14)
- Regime-adaptive: mean revert in chop, trend follow otherwise
- Fewer conflicting filters than #446 = more trades = better stats
- ATR 2.5x trailing stop protects in 2022-style crashes

Key improvements over #446:
- Lower timeframe (4h vs 12h) = more trade opportunities
- Looser CRSI thresholds (15/85 vs 10/90) = more entries
- Simpler regime detection (single CHOP threshold)
- Fallback entries to ensure minimum trade count
- Asymmetric sizing: 0.30 long, 0.25 short (bear market protection)

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 40-80 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_hma_12h_regime_simp_v1"
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
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Proven 75% win rate in research notes. Best for mean reversion entries.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak length
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    streak_delta = streak_s.diff()
    gain = streak_delta.where(streak_delta > 0, 0.0)
    loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_gain = gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_loss = loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    rs_streak = avg_gain / (avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # Component 3: Percent Rank of daily returns over 100 periods
    returns = close_s.pct_change()
    percent_rank = pd.Series(np.zeros(n))
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current = returns.iloc[i]
        if np.isnan(current):
            percent_rank.iloc[i] = 50.0
        else:
            rank = (window < current).sum()
            percent_rank.iloc[i] = (rank / rank_period) * 100.0
    
    crsi = (rsi_close + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    highest_high = high_s.rolling(window=period, min_periods=period).max().values
    lowest_low = low_s.rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF indicators (trend direction)
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    crsi_4h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        if np.isnan(crsi_4h[i]) or np.isnan(chop_4h[i]) or np.isnan(sma_200[i]):
            continue
        
        # === 12H TREND BIAS (primary direction filter) ===
        hma_12h_bullish = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        hma_12h_bearish = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # === 4H LOCAL TREND ===
        hma_4h_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === CHOPPINESS REGIME ===
        # Relaxed thresholds for more trades
        is_ranging = chop_4h[i] > 50.0
        is_trending = chop_4h[i] < 50.0
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_4h[i] < 20.0
        crsi_overbought = crsi_4h[i] > 80.0
        crsi_extreme_oversold = crsi_4h[i] < 15.0
        crsi_extreme_overbought = crsi_4h[i] > 85.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — SIMPLIFIED FOR MORE TRADES ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple conditions, any can trigger)
        if hma_12h_bullish:
            # Ranging: CRSI mean reversion
            if is_ranging and crsi_oversold:
                new_signal = LONG_SIZE
            # Trending: HMA alignment + CRSI pullback
            elif is_trending and hma_4h_bullish and crsi_4h[i] < 40.0:
                new_signal = LONG_SIZE
            # Extreme oversold (any regime)
            elif crsi_extreme_oversold:
                new_signal = LONG_SIZE
            # Simple: HMA bullish + CRSI low
            elif hma_4h_bullish and crsi_4h[i] < 30.0:
                new_signal = LONG_SIZE * 0.8
        
        # Also allow longs below SMA200 if CRSI extreme (bounce plays)
        if below_sma200 and crsi_extreme_oversold and hma_4h_bullish:
            if new_signal == 0.0:
                new_signal = LONG_SIZE * 0.6
        
        # SHORT ENTRIES
        if hma_12h_bearish:
            # Ranging: CRSI mean reversion
            if is_ranging and crsi_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Trending: HMA alignment + CRSI bounce
            elif is_trending and hma_4h_bearish and crsi_4h[i] > 60.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Extreme overbought (any regime)
            elif crsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Simple: HMA bearish + CRSI high
            elif hma_4h_bearish and crsi_4h[i] > 70.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
        
        # Also allow shorts above SMA200 if CRSI extreme (rejection plays)
        if above_sma200 and crsi_extreme_overbought and hma_4h_bearish:
            if new_signal == 0.0:
                new_signal = -SHORT_SIZE * 0.6
        
        # === FALLBACK ENTRIES (ensure minimum trade count) ===
        if not in_position and new_signal == 0.0:
            # Simple long: 12h bullish + 4h CRSI < 25
            if hma_12h_bullish and crsi_4h[i] < 25.0:
                new_signal = LONG_SIZE * 0.5
            # Simple short: 12h bearish + 4h CRSI > 75
            elif hma_12h_bearish and crsi_4h[i] > 75.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.5
        
        # === STOPLOSS CHECK (BEFORE exit logic) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit)
        if in_position and position_side > 0 and crsi_4h[i] > 85.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi_4h[i] < 15.0:
            new_signal = 0.0
        
        # Trend reversal exit
        if in_position and position_side > 0 and hma_12h_bearish and hma_4h_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_12h_bullish and hma_4h_bullish:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals