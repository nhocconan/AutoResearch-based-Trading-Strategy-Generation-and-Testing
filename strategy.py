#!/usr/bin/env python3
"""
Experiment #814: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime Switch

Hypothesis: Daily timeframe with weekly trend bias provides optimal signal quality
with minimal fee drag (20-50 trades/year). Connors RSI (CRSI) has proven 75% win
rate for mean reversion entries. Choppiness Index detects regime to switch between
mean-revert (chop) and trend-follow (trending) modes.

Key innovations:
1. 1w HMA(21) for ultra-long-term trend bias — only trade with weekly trend
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3. Choppiness Index(14) for regime: >61.8 = range (mean revert), <38.2 = trend
4. Asymmetric entries: long on CRSI<10 in bull, short on CRSI>90 in bear
5. ATR(14) 3.0x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.20, ±0.30 based on regime confidence

Entry conditions (designed for ≥20 trades/year on 1d):
- LONG: 1w HMA bull + CRSI<15 + (CHOP>55 OR price>SMA200)
- SHORT: 1w HMA bear + CRSI>85 + (CHOP>55 OR price<SMA200)
- Trend follow: CHOP<40 + HMA crossover confirmation

Target: Sharpe>0.45, trades>=20 train, trades>=3 test, DD>-35%
Timeframe: 1d
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
    streak = np.zeros(n)
    direction = np.zeros(n)  # 1=up, -1=down, 0=neutral
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            direction[i] = 1
            if direction[i-1] == 1:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif close[i] < close[i-1]:
            direction[i] = -1
            if direction[i-1] == -1:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            direction[i] = 0
            streak[i] = 0
    
    # Convert streak to RSI-like value (absolute streak length)
    abs_streak = np.abs(streak)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(streak_period, n):
        if abs_streak[i] >= streak_period:
            # Scale streak to 0-100 range
            streak_rsi[i] = min(100.0, abs_streak[i] * 25.0)
        else:
            streak_rsi[i] = abs_streak[i] * 25.0
    
    # Percent Rank - today's return vs last 100 days
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current_return = returns[i]
        rank = np.sum(window < current_return)
        percent_rank[i] = (rank / rank_period) * 100.0
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    valid_mask = (~np.isnan(rsi_short)) & (~np.isnan(streak_rsi)) & (~np.isnan(percent_rank))
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = range/choppy market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and tr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(tr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0  # neutral
    
    return choppiness

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):  # Start after 200 SMA + buffer
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(choppiness[i]) or np.isnan(sma_200[i]):
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
        
        # === HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        chop_range = choppiness[i] > 55.0  # Range/choppy market
        chop_trend = choppiness[i] < 45.0  # Trending market
        
        # === SMA200 FILTER ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_16[i-1]) and not np.isnan(hma_48[i-1]):
            hma_crossover_long = (hma_16[i-1] <= hma_48[i-1]) and (hma_16[i] > hma_48[i])
            hma_crossover_short = (hma_16[i-1] >= hma_48[i-1]) and (hma_16[i] < hma_48[i])
        
        # === HMA TREND ===
        hma_1d_bull = hma_16[i] > hma_48[i]
        hma_1d_bear = hma_16[i] < hma_48[i]
        
        # === CONNORS RSI EXTREMES (looser for more trades) ===
        crsi_oversold = crsi[i] < 20.0  # Was 10, now 20 for more trades
        crsi_overbought = crsi[i] > 80.0  # Was 90, now 80 for more trades
        crsi_extreme_oversold = crsi[i] < 12.0
        crsi_extreme_overbought = crsi[i] > 88.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entries (multiple confluence paths for trade generation)
        if htf_1w_bull:
            # Path 1: Mean reversion in choppy market
            if chop_range and crsi_oversold and price_above_sma200:
                if crsi_extreme_oversold:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # Path 2: Trend follow in trending market
            elif chop_trend and hma_crossover_long:
                desired_signal = SIZE_BASE
            
            # Path 3: HMA bull + CRSI pullback
            elif hma_1d_bull and crsi_oversold:
                desired_signal = SIZE_BASE
        
        # SHORT entries (multiple confluence paths for trade generation)
        elif htf_1w_bear:
            # Path 1: Mean reversion in choppy market
            if chop_range and crsi_overbought and price_below_sma200:
                if crsi_extreme_overbought:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            
            # Path 2: Trend follow in trending market
            elif chop_trend and hma_crossover_short:
                desired_signal = -SIZE_BASE
            
            # Path 3: HMA bear + CRSI pullback
            elif hma_1d_bear and crsi_overbought:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (3.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals