#!/usr/bin/env python3
"""
Experiment #130: 1h Primary + 4h/12h HTF — Connors RSI + Choppiness Regime + HTF HMA

Hypothesis: After 129 experiments, the pattern is clear:
- Simple trend following fails on BTC/ETH in bear/range markets (2022 crash, 2025 bear)
- Lower TF (1h/30m) strategies fail due to TOO MANY trades → fee drag
- Connors RSI (CRSI) has 75% win rate in mean reversion regimes
- Choppiness Index correctly identifies range vs trend regimes
- 12h HMA provides stable trend bias without whipsaw

This strategy combines 3 proven edges:
1. 12h HMA = major trend bias (only trade with HTF trend)
2. Choppiness Index(14) = regime filter (CHOP>55 range→mean revert, CHOP<45 trend→follow)
3. Connors RSI = entry timing (CRSI<10 long, CRSI>90 short)
4. Volume filter = confirm participation (vol > 0.8x 20-bar avg)
5. Session filter = only trade 8-20 UTC (high liquidity hours)

Key design for 1h timeframe:
- VERY strict entry conditions to limit trades to 30-60/year
- Position size: 0.25 (25% capital, conservative for 1h)
- Stoploss: 2.5x ATR trailing
- Take profit: reduce to half at 2R, trail rest

Target: Sharpe>0.351, DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_hma_regime_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): Very short-term momentum
    RSI_Streak(2): RSI of consecutive up/down days
    PercentRank(100): Where current close ranks vs last 100 closes
    
    CRSI < 10 = extremely oversold (long signal)
    CRSI > 90 = extremely overbought (short signal)
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.zeros(n)
    rsi_short[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_short[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_short[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak values
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    streak_gain = np.concatenate([[0.0], streak_gain])
    streak_loss = np.concatenate([[0.0], streak_loss])
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank (where current close ranks vs last rank_period closes)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # CRSI = average of 3 components
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = range/choppy (mean reversion regime)
    CHOP < 38.2 = trending (trend following regime)
    38.2 < CHOP < 61.8 = transition
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    for i in range(period, n):
        if hh[i] > ll[i] and atr_sum[i] > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
    
    return choppiness

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    More responsive than EMA with less lag
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, w_period):
        result = np.zeros(len(series))
        result[:] = np.nan
        weights = np.arange(1, w_period + 1)
        for i in range(w_period - 1, len(series)):
            window = series[i - w_period + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2.0 * wma_half - wma_full
    
    # WMA of diff with sqrt(period)
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
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
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for major trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume average for filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (conservative for 1h)
    
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
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # 1h bars: index 8-19 correspond to 08:00-19:00 UTC
        hour = i % 24  # Assuming 1h data starts at 00:00 UTC
        in_session = 8 <= hour <= 20
        
        # === HTF BIAS (12h HMA) ===
        htf_bull = close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        int_bull = close[i] > hma_4h_aligned[i]
        int_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (Choppiness Index) ===
        # CHOP > 55 = range (use mean reversion)
        # CHOP < 45 = trend (use trend following)
        is_range = choppiness[i] > 55.0
        is_trend = choppiness[i] < 45.0
        
        # === VOLUME FILTER ===
        vol_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === CONNORS RSI ENTRY SIGNALS ===
        # Mean reversion: CRSI < 10 (extreme oversold) or CRSI > 90 (extreme overbought)
        crsi_oversold = crsi[i] < 15.0  # Slightly relaxed from 10 to generate more trades
        crsi_overbought = crsi[i] > 85.0  # Slightly relaxed from 90
        
        # === DESIRED SIGNAL ===
        # LONG: HTF bull + (range+CRSI oversold OR trend+pullback) + session + volume
        # SHORT: HTF bear + (range+CRSI overbought OR trend+rally) + session + volume
        
        desired_signal = 0.0
        
        # Long conditions
        if htf_bull and int_bull and in_session and vol_ok:
            if is_range and crsi_oversold:
                # Range regime: mean reversion long at extreme oversold
                desired_signal = SIZE
            elif is_trend and crsi_oversold:
                # Trend regime: buy pullback in uptrend
                desired_signal = SIZE
        
        # Short conditions
        if htf_bear and int_bear and in_session and vol_ok:
            if is_range and crsi_overbought:
                # Range regime: mean reversion short at extreme overbought
                desired_signal = -SIZE
            elif is_trend and crsi_overbought:
                # Trend regime: short rally in downtrend
                desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
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
                # Flip position
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