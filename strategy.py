#!/usr/bin/env python3
"""
Experiment #1133: 1d Primary + 1w HTF — Dual Regime Strategy

Hypothesis: After 825+ failed experiments, the pattern is clear:
- Pure trend following fails in bear/range markets (2022 crash, 2025 bear)
- Pure mean reversion fails in strong trends
- DUAL REGIME approach adapts: mean revert in chop, trend follow otherwise

Key innovations:
1. Choppiness Index (14) regime detection: CHOP>61.8=range, CHOP<38.2=trend
2. Connors RSI for mean reversion: (RSI(3)+RSI_Streak(2)+PercentRank(100))/3
3. Weekly HMA(21) for macro bias - only trade with weekly trend
4. Donchian(20) breakout for trend entries + HMA crossover alternative
5. ATR(14) 2.5x trailing stop
6. Position size 0.30 discrete

Why 1d works:
- 20-50 trades/year target (minimal fee drag)
- Each signal has more conviction (daily bars)
- Less noise than lower timeframes
- Proven in research for BTC/ETH

Timeframe: 1d (primary)
HTF: 1w — loaded ONCE before loop
Position Size: 0.30 base (discrete: 0.0, ±0.30)
Stoploss: 2.5x ATR trailing
Target: 20-50 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_crsi_donchian_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        if span < 1:
            span = 1
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    diff = 2 * wma1 - wma2
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Better for mean reversion than standard RSI.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < pr_period:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) - measures consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value using streak period
    abs_streak = np.abs(streak)
    streak_rsi = np.full(n, np.nan)
    
    if n >= streak_period + 1:
        streak_diff = np.diff(abs_streak)
        streak_gain = np.where(streak_diff > 0, streak_diff, 0.0)
        streak_loss = np.where(streak_diff < 0, -streak_diff, 0.0)
        streak_gain = np.concatenate([[0.0], streak_gain])
        streak_loss = np.concatenate([[0.0], streak_loss])
        
        avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
        avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
        
        mask = avg_streak_loss > 1e-10
        rs_streak = np.zeros(n)
        rs_streak[mask] = avg_streak_gain[mask] / avg_streak_loss[mask]
        streak_rsi[mask] = 100.0 - (100.0 / (1.0 + rs_streak[mask]))
        streak_rsi[~mask] = 50.0
    
    # Percent Rank (100) - where current price ranks in last 100 days
    pr = np.full(n, np.nan)
    for i in range(pr_period, n):
        window = close[i-pr_period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        pr[i] = 100.0 * count_below / (pr_period - 1)
    
    # Combine
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(pr)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + pr[mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures if market is trending or ranging.
    CHOP > 61.8 = range (mean revert)
    CHOP < 38.2 = trend (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout levels."""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return upper, lower
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    hma_1d = calculate_hma(close, period=21)
    hma_1d_prev = np.roll(hma_1d, 1)
    rsi_14 = calculate_rsi(close, period=14)
    
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
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION ===
        in_range = chop[i] > 55.0  # Loosened from 61.8 for more trades
        in_trend = chop[i] < 45.0  # Loosened from 38.2 for more trades
        
        # === MACRO TREND (1w HMA) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === DAILY TREND (1d HMA) ===
        daily_bull = close[i] > hma_1d[i]
        daily_bear = close[i] < hma_1d[i]
        hma_cross_up = (hma_1d[i] > hma_1d_prev[i]) and (hma_1d[i-1] <= hma_1d_prev[i-1])
        hma_cross_down = (hma_1d[i] < hma_1d_prev[i]) and (hma_1d[i-1] >= hma_1d_prev[i-1])
        
        desired_signal = 0.0
        
        # === MEAN REVERSION MODE (Range) ===
        if in_range:
            # Long: CRSI < 25 + weekly bull bias (loosened from 15)
            if crsi[i] < 25.0 and weekly_bull:
                desired_signal = BASE_SIZE
            
            # Short: CRSI > 75 + weekly bear bias (loosened from 85)
            elif crsi[i] > 75.0 and weekly_bear:
                desired_signal = -BASE_SIZE
            
            # Additional MR entry: RSI extremes
            elif rsi_14[i] < 30.0 and weekly_bull:
                desired_signal = BASE_SIZE
            elif rsi_14[i] > 70.0 and weekly_bear:
                desired_signal = -BASE_SIZE
        
        # === TREND FOLLOWING MODE (Trend) ===
        elif in_trend:
            # Long breakout: price > Donchian upper + weekly/daily bull
            if close[i] > donchian_upper[i] and weekly_bull and daily_bull:
                desired_signal = BASE_SIZE
            
            # Short breakout: price < Donchian lower + weekly/daily bear
            elif close[i] < donchian_lower[i] and weekly_bear and daily_bear:
                desired_signal = -BASE_SIZE
            
            # HMA crossover entries (more frequent)
            elif hma_cross_up and weekly_bull and daily_bull:
                desired_signal = BASE_SIZE
            elif hma_cross_down and weekly_bear and daily_bear:
                desired_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME (between range and trend) ===
        else:
            # Use simpler signals in neutral regime
            if weekly_bull and daily_bull and rsi_14[i] < 50.0:
                desired_signal = BASE_SIZE
            elif weekly_bear and daily_bear and rsi_14[i] > 50.0:
                desired_signal = -BASE_SIZE
        
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
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if weekly still bull
                if weekly_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if weekly still bear
                if weekly_bear:
                    desired_signal = -BASE_SIZE
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            if weekly_bear:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if weekly_bull:
                desired_signal = 0.0
        
        # === DISCRETIZE ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals