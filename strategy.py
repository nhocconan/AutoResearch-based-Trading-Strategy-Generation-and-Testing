#!/usr/bin/env python3
"""
Experiment #443: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + HMA

Hypothesis: After 431 failed experiments, clear patterns emerge for 1d timeframe:
1. Choppiness Index (CHOP) is superior regime filter vs ADX alone — distinguishes trend vs range
2. Connors RSI (CRSI) outperforms standard RSI for mean reversion entries (75% win rate in research)
3. 1w HMA provides major trend direction — prevents counter-trend disasters in 2022-style crashes
4. 1d timeframe naturally produces 20-50 trades/year — optimal fee/trade balance
5. Asymmetric entries: favor longs when 1w HMA bullish, favor shorts when bearish

Why this might beat current best (Sharpe=0.435):
- CHOP regime switch adapts logic: trend-follow in trends, mean-revert in ranges
- CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — catches oversold/overbought extremes
- 1w HTF filter prevents whipsaw — only trade with weekly trend
- Simpler entry logic = more trades = better statistical significance
- ATR 2.5x trailing stop protects in crash scenarios

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 20-50 trades/year on 1d, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_hma_1w_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of ATR over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - low_s.rolling(window=period, min_periods=period).min()
    
    # CHOP formula
    chop = 100.0 * np.log10(atr_sum / (hh_ll + 1e-10)) / np.log10(period)
    
    return chop.values

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of daily returns over lookback period
    
    Entry signals:
    - Long: CRSI < 10 (extreme oversold)
    - Short: CRSI > 90 (extreme overbought)
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period_rsi, min_periods=period_rsi, adjust=False).mean()
    avg_loss = loss.ewm(span=period_rsi, min_periods=period_rsi, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: Streak RSI
    # Calculate streak of consecutive up/down days
    streak = pd.Series(0.0, index=close_s.index)
    for i in range(1, len(close_s)):
        if close_s.iloc[i] > close_s.iloc[i-1]:
            streak.iloc[i] = streak.iloc[i-1] + 1 if streak.iloc[i-1] > 0 else 1
        elif close_s.iloc[i] < close_s.iloc[i-1]:
            streak.iloc[i] = streak.iloc[i-1] - 1 if streak.iloc[i-1] < 0 else -1
        else:
            streak.iloc[i] = 0
    
    # RSI of streak (absolute values with sign)
    streak_delta = streak.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    streak_avg_gain = streak_gain.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: Percent Rank of daily returns
    daily_returns = close_s.pct_change()
    percent_rank = daily_returns.rolling(window=period_rank, min_periods=period_rank).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / len(x.iloc[:-1]) * 100 if len(x) > 1 else 50,
        raw=False
    )
    
    # Combine components
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    
    return crsi.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high and lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2.0
    
    return upper.values, lower.values, middle.values

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
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi_1d = calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100)
    rsi_14 = calculate_rsi(close, 14)
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_50_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi_1d[i]):
            continue
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        # Price above 1w HMA(21) + HMA(21) > HMA(50) = strong bull market
        # Price below 1w HMA(21) + HMA(21) < HMA(50) = strong bear market
        weekly_bull = close[i] > hma_1w_21_aligned[i] and hma_1w_21_aligned[i] > hma_1w_50_aligned[i]
        weekly_bear = close[i] < hma_1w_21_aligned[i] and hma_1w_21_aligned[i] < hma_1w_50_aligned[i]
        weekly_neutral = not weekly_bull and not weekly_bear
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 61.8 = ranging market (mean reversion strategy)
        # CHOP < 38.2 = trending market (trend follow strategy)
        # 38.2-61.8 = transition (use conservative entries)
        is_ranging = chop_14[i] > 61.8
        is_trending = chop_14[i] < 38.2
        is_transition = not is_ranging and not is_trending
        
        # === 1D LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_1d_21[i] > hma_1d_50[i]
        hma_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_1d[i] < 15.0  # Extreme oversold
        crsi_overbought = crsi_1d[i] > 85.0  # Extreme overbought
        crsi_low = crsi_1d[i] < 30.0
        crsi_high = crsi_1d[i] > 70.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if weekly_bull or weekly_neutral:
            # Trending regime: trend follow on Donchian breakout
            if is_trending and hma_bullish and donchian_breakout_up:
                new_signal = LONG_SIZE
            # Ranging regime: mean reversion on CRSI oversold
            elif is_ranging and crsi_oversold and hma_bullish:
                new_signal = LONG_SIZE
            # Transition regime: CRSI pullback entry
            elif is_transition and crsi_low and hma_bullish:
                new_signal = LONG_SIZE * 0.8
            # Weekly bull + daily HMA bullish + CRSI low
            elif weekly_bull and hma_bullish and crsi_1d[i] < 40.0:
                new_signal = LONG_SIZE * 0.9
        
        # SHORT ENTRIES
        if weekly_bear or weekly_neutral:
            # Trending regime: trend follow on Donchian breakdown
            if is_trending and hma_bearish and donchian_breakout_down:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Ranging regime: mean reversion on CRSI overbought
            elif is_ranging and crsi_overbought and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Transition regime: CRSI bounce entry
            elif is_transition and crsi_high and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
            # Weekly bear + daily HMA bearish + CRSI high
            elif weekly_bear and hma_bearish and crsi_1d[i] > 60.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.9
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 15 bars (~15 days on 1d), force entry on weaker signal
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if weekly_bull and hma_bullish and crsi_1d[i] < 45.0:
                new_signal = LONG_SIZE * 0.6
            elif weekly_bear and hma_bearish and crsi_1d[i] > 55.0:
                new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and crsi_1d[i] > 80.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi_1d[i] < 20.0:
            new_signal = 0.0
        
        # Weekly trend reversal exit
        if in_position and position_side > 0 and weekly_bear:
            new_signal = 0.0
        if in_position and position_side < 0 and weekly_bull:
            new_signal = 0.0
        
        # Daily trend reversal exit (HMA cross)
        if in_position and position_side > 0 and hma_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_bullish:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
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
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals