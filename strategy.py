#!/usr/bin/env python3
"""
Experiment #017: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime Switch

Hypothesis: Daily timeframe with weekly trend filter should work best for 
BTC/ETH in bear/range markets. Connors RSI captures short-term oversold/overbought 
conditions while Choppiness Index determines whether to trend-follow or mean-revert.

Key innovations:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More responsive than standard RSI, catches reversals faster
2. Choppiness Index regime detection:
   - CHOP > 61.8 = range (mean revert at CRSI extremes)
   - CHOP < 38.2 = trend (follow 1w HMA direction)
3. 1w HMA for HTF trend bias - only trade in direction of weekly trend
4. LOOSE CRSI thresholds (15/85) to ensure sufficient trades
5. Dual exit: time-based (5 days) + stoploss (2.5x ATR)

Entry Logic:
- Range regime (CHOP>61.8): Long CRSI<15, Short CRSI>85
- Trend regime (CHOP<38.2): Long if 1w HMA bull + CRSI<30, Short if 1w HMA bear + CRSI>70
- Size: 0.30 (discrete, minimizes fee churn)

Risk: 2.5x ATR trailing stop + 5-day max hold time
Target: Sharpe>0.3, trades>20/symbol train, >3/symbol test, DD>-40%
Timeframe: 1d (target 15-30 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines momentum, streak, and percentile rank
    More responsive than standard RSI for mean reversion entries
    """
    n = len(close)
    if n < rank_period + rsi_period:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3) - short-term momentum
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_short[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_short[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (2) - consecutive up/down days
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI on streak values
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            streak_rsi[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank (100) - where current return ranks vs past 100 days
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1]) / close[i-rank_period:i]
        current_return = returns[-1] if len(returns) > 0 else 0.0
        rank = np.sum(returns[:-1] < current_return) / max(len(returns) - 1, 1) * 100.0
        crsi[i] = (rsi_short[i] + streak_rsi[i] + rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
        else:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_hma(close, period=21):
    """Hull Moving Average - smooth and responsive"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA with half period
    wma_half = pd.Series(close).rolling(window=half, min_periods=half).mean().values
    # WMA with full period
    wma_full = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    
    # Raw HMA
    raw_hma = 2.0 * wma_half - wma_full
    
    # Smooth with sqrt period
    hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).mean().values
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
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
    
    # Calculate and align 1w HMA for HTF trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size
    
    # Position tracking for stoploss and time exit
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    MAX_HOLD_DAYS = 5  # Time-based exit
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 61.8  # Range-bound market
        is_trend = chop[i] < 38.2  # Trending market
        
        # === HTF TREND BIAS (1w HMA) ===
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_range:
            # Mean reversion in range - trade CRSI extremes
            if crsi[i] < 15.0:  # Oversold
                desired_signal = SIZE
            elif crsi[i] > 85.0:  # Overbought
                desired_signal = -SIZE
        elif is_trend:
            # Trend following - only trade with 1w trend on pullbacks
            if hma_1w_bull and crsi[i] < 30.0:  # Bull trend + pullback
                desired_signal = SIZE
            elif hma_1w_bear and crsi[i] > 70.0:  # Bear trend + rally
                desired_signal = -SIZE
        else:
            # Neutral regime - use moderate CRSI thresholds with HTF bias
            if hma_1w_bull and crsi[i] < 25.0:
                desired_signal = SIZE
            elif hma_1w_bear and crsi[i] > 75.0:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        time_exit_triggered = False
        
        if in_position:
            # Check time-based exit
            if i - entry_bar >= MAX_HOLD_DAYS:
                time_exit_triggered = True
            
            # Check trailing stop
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
                stop_price = highest_since_entry - 2.5 * entry_atr
                if close[i] < stop_price:
                    stoploss_triggered = True
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
                stop_price = lowest_since_entry + 2.5 * entry_atr
                if close[i] > stop_price:
                    stoploss_triggered = True
        
        if stoploss_triggered or time_exit_triggered:
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
                entry_bar = i
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                entry_bar = i
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
                entry_bar = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals