#!/usr/bin/env python3
"""
Experiment #237: 1d Primary + 1w HTF — Connors RSI + Donchian Breakout + Choppiness Regime

Hypothesis: Daily timeframe with weekly macro bias can capture major trends while using
Connors RSI for optimal entry timing. Choppiness Index filters between mean-reversion
(choppy) and trend-following (directional) regimes. Donchian breakouts confirm momentum.

Key design:
1. Connors RSI (CRSI) for entry timing: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
2. Choppiness Index(14) regime filter: >61.8 = range (mean revert), <38.2 = trend
3. Donchian(20) breakout confirmation for trend entries
4. 1w HMA(21) for macro trend bias (aligns with major market direction)
5. ATR(14) 2.5x trailing stoploss
6. Discrete position sizing: 0.0, ±0.25, ±0.30

TARGET: 15-30 trades/year on 1d, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
Why this might work: CRSI has 75% win rate in research, Choppiness filters bad regimes,
1w HTF prevents counter-trend trades during major moves.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_donchian_chop_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

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

def calculate_crsi(close):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down day count
    PercentRank: percentile rank of current close in last 100 closes
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - very short term momentum
    rsi_3 = calculate_rsi(close, period=3)
    
    # RSI Streak (2) - RSI of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI of streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=2, min_periods=2, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=2, min_periods=2, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # PercentRank(100) - where current price ranks in last 100
    percent_rank = np.zeros(n)
    for i in range(100, n):
        window = close[i-99:i+1]  # last 100 including current
        rank = np.sum(window < close[i]) / 100.0 * 100.0
        percent_rank[i] = rank
    percent_rank[:100] = 50.0  # fill initial values
    
    # Combine into CRSI
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range market (mean reversion favorable)
    CHOP < 38.2 = trending market (trend following favorable)
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate 1w HMA for macro trend (aligned properly)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    
    for i in range(150, n):  # Need 150+ for CRSI percentrank + Donchian
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HTF MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        macro_bullish = price_above_hma_1w
        macro_bearish = price_below_hma_1w
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range/mean-reversion regime
        is_trending = chop[i] < 45.0  # Trend-following regime
        # Neutral zone: 45-55
        
        # === CRSI EXTREMES (mean reversion signals) ===
        crsi_oversold = crsi[i] < 15.0  # Very oversold
        crsi_overbought = crsi[i] > 85.0  # Very overbought
        crsi_neutral_low = 15.0 <= crsi[i] <= 40.0
        crsi_neutral_high = 60.0 <= crsi[i] <= 85.0
        
        # === DONCHIAN BREAKOUT (trend confirmation) ===
        donchian_breakout_long = close[i] >= donchian_upper[i-1]  # Break above previous upper
        donchian_breakout_short = close[i] <= donchian_lower[i-1]  # Break below previous lower
        
        # === DETERMINE DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY CONDITIONS
        if macro_bullish:
            # Trend-following long: trending regime + Donchian breakout + CRSI not overbought
            if is_trending and donchian_breakout_long and crsi[i] < 75.0:
                desired_signal = POSITION_SIZE_FULL
            # Mean-reversion long: choppy regime + CRSI oversold
            elif is_choppy and crsi_oversold:
                desired_signal = POSITION_SIZE_HALF
            # Neutral regime + CRSI recovering from oversold
            elif not is_trending and not is_choppy and crsi_neutral_low and crsi[i] > crsi[i-1]:
                desired_signal = POSITION_SIZE_HALF
        
        # SHORT ENTRY CONDITIONS
        elif macro_bearish:
            # Trend-following short: trending regime + Donchian breakout + CRSI not oversold
            if is_trending and donchian_breakout_short and crsi[i] > 25.0:
                desired_signal = -POSITION_SIZE_FULL
            # Mean-reversion short: choppy regime + CRSI overbought
            elif is_choppy and crsi_overbought:
                desired_signal = -POSITION_SIZE_HALF
            # Neutral regime + CRSI falling from overbought
            elif not is_trending and not is_choppy and crsi_neutral_high and crsi[i] < crsi[i-1]:
                desired_signal = -POSITION_SIZE_HALF
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (take profit on mean reversion) ===
        if in_position and position_side > 0 and crsi_overbought:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi_oversold:
            desired_signal = 0.0
        
        # === MACRO REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bearish and crsi[i] > 60.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bullish and crsi[i] < 40.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if thesis still valid ===
        if in_position and desired_signal == 0.0:
            if position_side > 0 and macro_bullish and crsi[i] < 80.0:
                desired_signal = POSITION_SIZE_HALF
            elif position_side < 0 and macro_bearish and crsi[i] > 20.0:
                desired_signal = -POSITION_SIZE_HALF
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals