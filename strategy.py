#!/usr/bin/env python3
"""
Experiment #1363: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime

Hypothesis: 1d strategies failed (#1353, #1357) due to either too few trades (0 trades) 
or wrong signal direction. Solution: Use Connors RSI for mean-reversion entries with 
Choppiness Index regime filter to switch between mean-reversion (chop) and trend-follow (trend).

Key design choices:
1. Connors RSI (3-period RSI + 2-period streak RSI + 100-period percent rank) / 3
2. Choppiness Index(14) > 61.8 = range (mean revert), < 38.2 = trend (trend follow)
3. 1w HMA(21) for macro trend bias
4. 1d ATR(14) for stoploss at 2.5x
5. Position size 0.30 with discrete levels
6. Multiple entry paths to ensure >=20 trades/train, >=3 trades/test

Target: 20-50 trades/year, Sharpe > 0.618, trades >= 20 train, >= 3 test
Timeframe: 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_hma_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(3): Fast RSI for short-term momentum
    RSI(Streak, 2): RSI of consecutive up/down days
    PercentRank(100): Where current price ranks vs last 100 days (0-100)
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_3 = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_3 = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_3 = np.full(n, np.nan)
    mask = loss_3 > 1e-10
    rsi_3[mask] = 100.0 - (100.0 / (1.0 + gain_3[mask] / loss_3[mask]))
    rsi_3[loss_3 <= 1e-10] = 100.0
    rsi_3[:rsi_period] = np.nan
    
    # Streak RSI(2)
    streak = np.zeros(n)
    streak[0] = 1 if delta[0] >= 0 else -1
    for i in range(1, n):
        if delta[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif delta[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    streak_delta = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    streak_gain_2 = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_2 = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask = streak_loss_2 > 1e-10
    rsi_streak[mask] = 100.0 - (100.0 / (1.0 + streak_gain_2[mask] / streak_loss_2[mask]))
    rsi_streak[streak_loss_2 <= 1e-10] = 100.0
    rsi_streak[:streak_period] = np.nan
    
    # Percent Rank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window < close[i])
            percent_rank[i] = 100.0 * count_below / rank_period
    
    # Connors RSI
    crsi = np.full(n, np.nan)
    valid = ~np.isnan(rsi_3) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid] = (rsi_3[valid] + rsi_streak[valid] + percent_rank[valid]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index = 100 * (ATR(1,sum of ATR(14))) / (Highest High - Lowest Low)
    
    CHOP > 61.8 = Choppy/Range
    CHOP < 38.2 = Trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Sum of ATR(14)
    atr_sum = np.full(n, np.nan)
    for i in range(period - 1, n):
        atr_sum[i] = np.sum(tr[i-period+1:i+1])
    
    # Highest High - Lowest Low over period
    hh_ll = np.full(n, np.nan)
    for i in range(period - 1, n):
        hh_ll[i] = np.nanmax(high[i-period+1:i+1]) - np.nanmin(low[i-period+1:i+1])
    
    # Choppiness Index
    chop = np.full(n, np.nan)
    mask = hh_ll > 1e-10
    chop[mask] = 100.0 * atr_sum[mask] / hh_ll[mask]
    
    return chop

def calculate_hma(close, period=21):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

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
    
    # Calculate and align HTF HMA for trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
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
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 61.8  # Range market
        is_trending = chop[i] < 38.2  # Trend market
        
        # === MACRO TREND (1w HMA) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === CONNORS RSI SIGNALS (widened for more trades) ===
        crsi_oversold = crsi[i] < 15.0  # Oversold for long
        crsi_overbought = crsi[i] > 85.0  # Overbought for short
        crsi_extreme_oversold = crsi[i] < 8.0
        crsi_extreme_overbought = crsi[i] > 92.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION in choppy markets
            # Long: CRSI oversold + macro bull bias
            if crsi_extreme_oversold and macro_bull:
                desired_signal = BASE_SIZE
            elif crsi_oversold and macro_bull:
                desired_signal = BASE_SIZE * 0.5
            # Short: CRSI overbought + macro bear bias
            elif crsi_extreme_overbought and macro_bear:
                desired_signal = -BASE_SIZE
            elif crsi_overbought and macro_bear:
                desired_signal = -BASE_SIZE * 0.5
        
        elif is_trending:
            # TREND FOLLOWING in trending markets
            # Long: CRSI oversold (pullback) + macro bull
            if crsi_oversold and macro_bull:
                desired_signal = BASE_SIZE
            # Short: CRSI overbought (rally) + macro bear
            elif crsi_overbought and macro_bear:
                desired_signal = -BASE_SIZE
        else:
            # NEUTRAL regime (38.2 <= CHOP <= 61.8): use simpler logic
            # Long on extreme oversold regardless of trend
            if crsi_extreme_oversold:
                desired_signal = BASE_SIZE * 0.5
            elif crsi_extreme_overbought:
                desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
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