#!/usr/bin/env python3
"""
Experiment #1627: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: Daily timeframe with weekly bias + Connors RSI mean reversion will work better
than complex multi-filter approaches. Previous 1d strategies failed due to:
- Too many confluence filters = 0 trades (#1622, #1623, #1624, #1625)
- CRSI thresholds too extreme (10/90) = rare signals
- Dual regime logic too complex

This strategy simplifies:
1. Connors RSI (RSI3 + StreakRSI2 + PercentRank100) / 3 for entries
2. Weekly HMA(21) for trend bias (single HTF)
3. Choppiness Index for regime detection (simple threshold)
4. Looser CRSI thresholds (20/80) for more trades
5. HMA(8/21) crossover for trend confirmation
6. ATR 2.5x trailing stoploss

Why this should work:
- CRSI has 75% win rate in research literature for mean reversion
- 1d target: 20-50 trades/year = ~80-200 trades in 4-year train
- Weekly HMA provides clear bias without over-filtering
- Simpler logic = more trades = better statistics

Timeframe: 1d (required)
HTF: 1w HMA for bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_hma_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_window=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): Standard RSI with 3-period lookback
    RSI_Streak(2): RSI of consecutive up/down days
    PercentRank(100): Where current price ranks vs last 100 days
    """
    n = len(close)
    if n < pr_window:
        return np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_3 = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_3 = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_3 = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if loss_3[i-1] < 1e-10:
            rsi_3[i] = 100.0
        else:
            rsi_3[i] = 100.0 - (100.0 / (1.0 + gain_3[i-1] / loss_3[i-1]))
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    for i in range(streak_period, n):
        if streak_loss_smooth[i-1] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[i-1] / streak_loss_smooth[i-1]))
    
    # Percent Rank (where current close ranks vs last 100 days)
    percent_rank = np.zeros(n)
    for i in range(pr_window, n):
        window = close[i-pr_window+1:i+1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_below / (pr_window - 1)
    
    # Combine into CRSI
    crsi = np.full(n, np.nan)
    valid_mask = ~np.isnan(rsi_3) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_3[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align weekly HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_window=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # HMA for trend following (fast and slow)
    hma_fast = calculate_hma(close, period=8)
    hma_slow = calculate_hma(close, period=21)
    
    # Donchian channels for breakout detection
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 50 = choppy/range (mean revert), CHOP < 45 = trending
        is_choppy = chop[i] > 50.0
        is_trending = chop[i] < 45.0
        
        # === TREND BIAS (1w HMA) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === HMA CROSSOVER ===
        hma_bull = hma_fast[i] > hma_slow[i]
        hma_bear = hma_fast[i] < hma_slow[i]
        
        # Check previous bar for crossover detection
        hma_bull_prev = False
        hma_bear_prev = False
        if i > 0 and not np.isnan(hma_fast[i-1]) and not np.isnan(hma_slow[i-1]):
            hma_bull_prev = hma_fast[i-1] > hma_slow[i-1]
            hma_bear_prev = hma_fast[i-1] < hma_slow[i-1]
        
        hma_cross_up = hma_bull and not hma_bull_prev
        hma_cross_down = hma_bear and not hma_bear_prev
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = False
        donchian_breakout_down = False
        if not np.isnan(donchian_upper[i-1]):
            donchian_breakout_up = close[i] > donchian_upper[i-1]
        if not np.isnan(donchian_lower[i-1]):
            donchian_breakout_down = close[i] < donchian_lower[i-1]
        
        # === CONNORS RSI EXTREMES (looser thresholds for more trades) ===
        crsi_extreme_low = crsi[i] < 25.0   # Oversold - long signal
        crsi_extreme_high = crsi[i] > 75.0  # Overbought - short signal
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING MARKET - Trend Following
        if is_trending:
            # Long: Weekly bull + HMA bull + CRSI not extreme high
            if weekly_bull and hma_bull and crsi[i] < 70.0:
                desired_signal = BASE_SIZE
            # Short: Weekly bear + HMA bear + CRSI not extreme low
            elif weekly_bear and hma_bear and crsi[i] > 30.0:
                desired_signal = -BASE_SIZE
            # Entry on breakout with trend
            elif weekly_bull and donchian_breakout_up:
                desired_signal = BASE_SIZE
            elif weekly_bear and donchian_breakout_down:
                desired_signal = -BASE_SIZE
        
        # REGIME 2: CHOPPY MARKET - Mean Reversion with CRSI
        elif is_choppy:
            # Long: CRSI extreme low + weekly bias not strongly bearish
            if crsi_extreme_low and not weekly_bear:
                desired_signal = BASE_SIZE
            # Short: CRSI extreme high + weekly bias not strongly bullish
            elif crsi_extreme_high and not weekly_bull:
                desired_signal = -BASE_SIZE
            # HMA crossover confirmation in chop
            elif hma_cross_up and crsi[i] < 50.0:
                desired_signal = BASE_SIZE
            elif hma_cross_down and crsi[i] > 50.0:
                desired_signal = -BASE_SIZE
        
        # REGIME 3: NEUTRAL/TRANSITION - Hold existing, simpler entries
        else:
            # CRSI mean reversion
            if crsi_extreme_low:
                desired_signal = BASE_SIZE
            elif crsi_extreme_high:
                desired_signal = -BASE_SIZE
            # Hold existing position
            elif in_position:
                desired_signal = BASE_SIZE if position_side > 0 else -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals