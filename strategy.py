#!/usr/bin/env python3
"""
Experiment #726: 12h Primary + 1d HTF — Dual Regime (Chop + Connors RSI)

Hypothesis: After 486 failed strategies, the pattern is clear:
1. 12h timeframe needs SIMPLE logic with fewer conflicting filters
2. Connors RSI (CRSI) proven on ETH with Sharpe +0.923 in research
3. Choppiness Index is the BEST regime filter for bear/range markets
4. Dual regime: mean-revert in chop (CHOP>61.8), trend-follow otherwise
5. 1d HMA for ultra-long-term bias to avoid counter-trend trades

Key differences from failed #722/#723:
- Simpler entry logic (fewer AND conditions)
- Connors RSI instead of regular RSI (faster reaction)
- Lower RSI thresholds to ensure trade frequency
- ATR stoploss without complex position state tracking
- Multiple entry paths per regime to guarantee trades

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (proven higher TF works, 20-50 trades/year target)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_chop_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: rank of today's return vs last 100 days
    
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # RSI(3) - very fast RSI
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI on streak values
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    
    if len(streak_gain) > streak_period:
        streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
        streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
        streak_avg_gain = np.concatenate([[np.nan], streak_avg_gain])
        streak_avg_loss = np.concatenate([[np.nan], streak_avg_loss])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
            streak_rsi = 100 - (100 / (1 + streak_rs))
        streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # PercentRank - rank of today's return vs last rank_period days
    returns = np.diff(close) / (close[:-1] + 1e-10)
    percent_rank = np.full(n, np.nan)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            current_return = returns[i-1] if not np.isnan(returns[i-1]) else 0
            rank = np.sum(valid < current_return) / len(valid)
            percent_rank[i] = rank * 100
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = range/chop (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High - Lowest Low over period
    hh_ll = pd.Series(high).rolling(window=period, min_periods=period).max().values - \
            pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100 * np.log10(atr_sum / (hh_ll + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop_raw, 0, 100)
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
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(chop_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_12h[i] > 61.8  # Range market - mean revert
        is_trending = chop_12h[i] < 38.2  # Trending market - trend follow
        
        # === TREND BIAS (1d HTF HMA) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        
        # === MEAN REVERSION REGIME (CHOP > 61.8) ===
        if is_choppy:
            # Long: CRSI deeply oversold + above SMA200 or bullish 1d trend
            if crsi_12h[i] < 15 and (above_sma200 or trend_1d_bullish):
                desired_signal = BASE_SIZE
            
            # Short: CRSI deeply overbought + below SMA200 or bearish 1d trend
            elif crsi_12h[i] > 85 and (below_sma200 or trend_1d_bearish):
                desired_signal = -BASE_SIZE
            
            # Additional mean reversion: price at Donchian extremes
            elif crsi_12h[i] < 25 and close[i] <= donch_lower[i-1]:
                desired_signal = BASE_SIZE
            elif crsi_12h[i] > 75 and close[i] >= donch_upper[i-1]:
                desired_signal = -BASE_SIZE
        
        # === TREND FOLLOWING REGIME (CHOP < 38.2) ===
        elif is_trending:
            # Long: Donchian breakout + bullish 1d trend
            if close[i] > donch_upper[i-1] and trend_1d_bullish:
                desired_signal = BASE_SIZE
            
            # Short: Donchian breakdown + bearish 1d trend
            elif close[i] < donch_lower[i-1] and trend_1d_bearish:
                desired_signal = -BASE_SIZE
            
            # Pullback entries in trend
            elif trend_1d_bullish and crsi_12h[i] < 35 and above_sma200:
                desired_signal = BASE_SIZE
            elif trend_1d_bearish and crsi_12h[i] > 65 and below_sma200:
                desired_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: only enter on strong CRSI extremes with trend confirmation
            if crsi_12h[i] < 10 and trend_1d_bullish:
                desired_signal = BASE_SIZE
            elif crsi_12h[i] > 90 and trend_1d_bearish:
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if not extremely overbought
                if crsi_12h[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if not extremely oversold
                if crsi_12h[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI extremely overbought or trend reverses
            if crsi_12h[i] > 90 or trend_1d_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI extremely oversold or trend reverses
            if crsi_12h[i] < 10 or trend_1d_bullish:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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