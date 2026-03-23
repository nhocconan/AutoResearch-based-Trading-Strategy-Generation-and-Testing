#!/usr/bin/env python3
"""
Experiment #561: 4h Primary + 1d HTF — Choppiness Regime + Connors RSI + Donchian Breakout

Hypothesis: After 400+ experiments, regime-adaptive strategies outperform static ones.
Choppiness Index (CHOP) detects market state: CHOP>61.8=range (mean revert), CHOP<38.2=trend.
Connors RSI (CRSI) excels at mean reversion entries (75% win rate documented).
Donchian breakouts capture trend momentum with HTF bias filter.

Key improvements over #491:
1. Choppiness Index regime detection (proven ETH Sharpe +0.923)
2. Connors RSI instead of standard RSI (RSI(3)+RSI_Streak+PercentRank)/3
3. Regime-adaptive: mean revert in chop, trend follow otherwise
4. Relaxed entry thresholds for trade generation (target 30+ trades/train)
5. 1d HMA for major trend bias (simpler than KAMA, proven stable)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_crsi_donchian_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    Measures market choppiness vs trending.
    CHOP > 61.8 = rangebound (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI).
    Composite of 3 components for mean reversion signals.
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    Long: CRSI < 10 (oversold)
    Short: CRSI > 90 (overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # Component 1: RSI(3) on close
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    gain_s = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI on Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] >= 0 else 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak (treat absolute streak as "price")
    streak_gain = np.zeros(n)
    streak_loss = np.zeros(n)
    streak_delta = np.diff(streak)
    streak_gain[1:] = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss[1:] = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    streak_gain_s = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_s = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = streak_gain_s / (streak_loss_s + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: Percent Rank of daily returns over 100 periods
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i - rank_period:i + 1])
        if len(returns) > 0 and len(returns) == rank_period:
            current_return = close[i] - close[i - 1]
            rank = np.sum(returns < current_return)
            percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA).
    Reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, wma_period):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, wma_period + 1)
        for i in range(wma_period - 1, len(series)):
            window = series[i - wma_period + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # Raw HMA
    raw = np.zeros(n)
    for i in range(n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            raw[i] = 2.0 * wma_half[i] - wma_full[i]
        else:
            raw[i] = np.nan
    
    # Final WMA on raw
    hma = wma(raw, sqrt_period)
    
    return hma

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel.
    Upper = Highest High over n periods
    Lower = Lowest Low over n periods
    """
    n = len(close) if 'close' in dir() else len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF indicators (1d HMA for major trend bias)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(chop_4h[i]):
            continue
        if np.isnan(crsi_4h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = rangebound (mean revert)
        # CHOP < 38.2 = trending (trend follow)
        # 38.2 - 61.8 = neutral (use HTF bias)
        is_choppy = chop_4h[i] > 55.0  # Relaxed threshold for more signals
        is_trending = chop_4h[i] < 45.0  # Relaxed threshold
        
        # === HTF MAJOR TREND BIAS (1d HMA) ===
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # === MEAN REVERSION MODE (Choppy market) ===
        if is_choppy and not is_trending:
            # Long: CRSI < 20 (oversold) + HTF not strongly bearish
            if crsi_4h[i] < 20.0 and not htf_bearish:
                desired_signal = SIZE_LONG
            # Short: CRSI > 80 (overbought) + HTF not strongly bullish
            elif crsi_4h[i] > 80.0 and not htf_bullish:
                desired_signal = -SIZE_SHORT
        
        # === TREND FOLLOWING MODE (Trending market) ===
        elif is_trending and not is_choppy:
            # Long: Breakout above Donchian + HTF bullish
            if close[i] > donchian_upper[i] and htf_bullish:
                desired_signal = SIZE_LONG
            # Short: Breakdown below Donchian + HTF bearish
            elif close[i] < donchian_lower[i] and htf_bearish:
                desired_signal = -SIZE_SHORT
        
        # === NEUTRAL MODE (Use HTF bias only) ===
        else:
            # Long: HTF bullish + CRSI not overbought
            if htf_bullish and crsi_4h[i] < 70.0:
                desired_signal = SIZE_LONG
            # Short: HTF bearish + CRSI not oversold
            elif htf_bearish and crsi_4h[i] > 30.0:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF still bullish or CRSI not overbought
                if htf_bullish or crsi_4h[i] < 75.0:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF still bearish or CRSI not oversold
                if htf_bearish or crsi_4h[i] > 25.0:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
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