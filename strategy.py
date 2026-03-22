#!/usr/bin/env python3
"""
Experiment #473: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: After 472 experiments, clear pattern emerges for 1d timeframe:
1. 1d has proven potential (research shows ETH Sharpe +0.923 with CRSI+CHOP)
2. Choppiness Index regime detection is critical for bear/range markets (2025 test period)
3. Connors RSI (not standard RSI) catches reversals better with 75% win rate
4. 1w HMA provides clean major trend filter without over-complication
5. Simpler logic = more trades (critical: need >=30 trades/symbol on train)

Why this might beat current best (Sharpe=0.435):
- CRSI extremes (15/85) trigger more frequently than RSI(14) extremes
- Choppiness regime switch adapts to 2025 bear/range market
- 1w trend filter prevents counter-trend trades in strong moves
- ATR 2.5x trailing stop protects in 2022-style crashes
- 1d timeframe = 20-50 trades/year = 80-200 trades on 4yr train

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 20-50 trades/year, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_crsi_hma_1w_v1"
timeframe = "1d"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Proven 75% win rate in research notes. Best for mean reversion entries.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on close
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI on streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if pd.isna(delta.iloc[i]):
            streak[i] = streak[i-1]
        elif delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    streak_delta = streak_s.diff()
    gain_streak = streak_delta.where(streak_delta > 0, 0.0)
    loss_streak = -streak_delta.where(streak_delta < 0, 0.0)
    avg_gain_streak = gain_streak.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_loss_streak = loss_streak.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    rs_streak = avg_gain_streak / (avg_loss_streak + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # Component 3: Percent Rank of daily returns over 100 periods
    returns = close_s.pct_change()
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current = returns.iloc[i]
        if np.isnan(current):
            percent_rank[i] = 50.0
        else:
            rank = (window < current).sum()
            percent_rank[i] = (rank / rank_period) * 100.0
    
    crsi = (rsi_close.values + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_50 = calculate_hma(df_1w['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi_1d = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_50_aligned[i]):
            continue
        if np.isnan(crsi_1d[i]) or np.isnan(chop_1d[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1w_21_aligned[i]
        bear_regime = close[i] < hma_1w_21_aligned[i]
        trend_strong_bull = hma_1w_21_aligned[i] > hma_1w_50_aligned[i]
        trend_strong_bear = hma_1w_21_aligned[i] < hma_1w_50_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        is_ranging = chop_1d[i] > 55.0
        is_trending = chop_1d[i] < 45.0
        
        # === CONNORS RSI SIGNALS (relaxed for frequency) ===
        crsi_oversold = crsi_1d[i] < 25.0
        crsi_overbought = crsi_1d[i] > 75.0
        crsi_extreme_oversold = crsi_1d[i] < 15.0
        crsi_extreme_overbought = crsi_1d[i] > 85.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_high = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_low = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === ENTRY LOGIC — DUAL REGIME WITH MULTIPLE TRIGGERS ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple conditions for frequency)
        if bull_regime and crsi_oversold:
            new_signal = LONG_SIZE
        elif crsi_extreme_oversold and above_sma200:
            new_signal = LONG_SIZE
        elif is_ranging and crsi_1d[i] < 20.0:
            new_signal = LONG_SIZE
        elif is_trending and bull_regime and crsi_1d[i] < 35.0:
            new_signal = LONG_SIZE
        elif breakout_high and bull_regime and crsi_1d[i] < 50.0:
            new_signal = LONG_SIZE
        elif above_sma200 and trend_strong_bull and crsi_1d[i] < 30.0:
            new_signal = LONG_SIZE
        
        # SHORT ENTRIES (multiple conditions for frequency)
        if new_signal == 0.0:
            if bear_regime and crsi_overbought:
                new_signal = -SHORT_SIZE
            elif crsi_extreme_overbought and below_sma200:
                new_signal = -SHORT_SIZE
            elif is_ranging and crsi_1d[i] > 80.0:
                new_signal = -SHORT_SIZE
            elif is_trending and bear_regime and crsi_1d[i] > 65.0:
                new_signal = -SHORT_SIZE
            elif breakout_low and bear_regime and crsi_1d[i] > 50.0:
                new_signal = -SHORT_SIZE
            elif below_sma200 and trend_strong_bear and crsi_1d[i] > 70.0:
                new_signal = -SHORT_SIZE
        
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
        
        # === TAKE PROFIT / EXIT CONDITIONS ===
        if in_position and position_side > 0 and crsi_1d[i] > 80.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi_1d[i] < 20.0:
            new_signal = 0.0
        
        # Regime flip exit
        if in_position and position_side > 0 and bear_regime and trend_strong_bear:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime and trend_strong_bull:
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