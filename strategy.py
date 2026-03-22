#!/usr/bin/env python3
"""
Experiment #461: 4h Primary + 1d/1w HTF — Dual Regime (Trend/Mean-Revert) + Connors RSI

Hypothesis: After analyzing 460 failed experiments, clear pattern emerges for 4h:
1. 1w HMA provides ultra-slow trend bias (avoids 2022-style whipsaws)
2. 1d HMA gives intermediate trend confirmation
3. 4h Choppiness Index switches between trend-follow and mean-revert modes
4. Connors RSI (proven 75% win rate) for precise entry timing
5. Donchian breakout for trend continuation in trending regimes
6. Dual-regime approach: mean-revert in chop, trend-follow otherwise

Why this might beat current best (Sharpe=0.435):
- 4h TF balances trade frequency (20-50/year) with signal quality
- 1w HTF filter prevents counter-trend trades in major moves
- Connors RSI catches reversals better than standard RSI(14)
- Choppiness regime adapts automatically to market conditions
- ATR 2.5x trailing stop protects in crashes

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 20-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_crsi_donchian_1d1w_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (ultra-slow trend bias)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d HTF indicators (intermediate trend)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi_4h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(crsi_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W MAJOR TREND (ultra-slow bias) ===
        bull_1w = close[i] > hma_1w_21_aligned[i]
        bear_1w = close[i] < hma_1w_21_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        bull_1d = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        bear_1d = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 55 = ranging (mean reversion mode)
        # CHOP < 45 = trending (trend follow mode)
        is_ranging = chop_4h[i] > 55.0
        is_trending = chop_4h[i] < 45.0
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_4h[i] < 20.0
        crsi_overbought = crsi_4h[i] > 80.0
        crsi_extreme_oversold = crsi_4h[i] < 10.0
        crsi_extreme_overbought = crsi_4h[i] > 90.0
        crsi_moderate_oversold = crsi_4h[i] < 30.0
        crsi_moderate_overbought = crsi_4h[i] > 70.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ENTRY LOGIC — DUAL REGIME ===
        new_signal = 0.0
        
        # LONG ENTRIES
        if bull_1w:  # Only long in 1w bull regime
            # MEAN REVERSION MODE (ranging market)
            if is_ranging:
                if crsi_extreme_oversold:
                    new_signal = LONG_SIZE
                elif crsi_oversold and price_above_1d_hma:
                    new_signal = LONG_SIZE * 0.8
                elif crsi_moderate_oversold and bull_1d:
                    new_signal = LONG_SIZE * 0.6
            
            # TREND FOLLOW MODE (trending market)
            elif is_trending:
                if donchian_breakout_long and bull_1d:
                    new_signal = LONG_SIZE
                elif bull_1d and crsi_moderate_oversold:
                    new_signal = LONG_SIZE * 0.7
                elif price_above_1d_hma and crsi_oversold:
                    new_signal = LONG_SIZE * 0.6
            
            # GENERAL LONG (works in any regime)
            if crsi_extreme_oversold and bull_1d:
                if new_signal == 0.0:
                    new_signal = LONG_SIZE * 0.5
        
        # SHORT ENTRIES
        if bear_1w:  # Only short in 1w bear regime
            # MEAN REVERSION MODE (ranging market)
            if is_ranging:
                if crsi_extreme_overbought:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE
                elif crsi_overbought and price_below_1d_hma:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE * 0.8
                elif crsi_moderate_overbought and bear_1d:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE * 0.6
            
            # TREND FOLLOW MODE (trending market)
            elif is_trending:
                if donchian_breakout_short and bear_1d:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE
                elif bear_1d and crsi_moderate_overbought:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE * 0.7
                elif price_below_1d_hma and crsi_overbought:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE * 0.6
            
            # GENERAL SHORT (works in any regime)
            if crsi_extreme_overbought and bear_1d:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.5
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # Relax entry if no position and market conditions are clear
        if not in_position and new_signal == 0.0:
            # Long: simpler conditions
            if bull_1w and bull_1d and crsi_4h[i] < 35.0:
                new_signal = LONG_SIZE * 0.5
            # Short: simpler conditions
            elif bear_1w and bear_1d and crsi_4h[i] > 65.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.5
        
        # === STOPLOSS CHECK (BEFORE exit logic - CRITICAL) ===
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
        # CRSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and crsi_4h[i] > 85.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi_4h[i] < 15.0:
            new_signal = 0.0
        
        # 1W regime flip exit (major trend reversal)
        if in_position and position_side > 0 and bear_1w:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_1w:
            new_signal = 0.0
        
        # 1D trend reversal exit
        if in_position and position_side > 0 and bear_1d and price_below_1d_hma:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_1d and price_above_1d_hma:
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
                # Position flip
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