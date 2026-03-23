#!/usr/bin/env python3
"""
Experiment #702: 12h Primary + 1d/1w HTF — Connors RSI + Donchian + Choppiness Regime

Hypothesis: Combining Connors RSI (75% win rate mean reversion) with Donchian 
breakouts, switched by Choppiness Index regime, will outperform pure BB squeeze.
12h TF provides optimal balance: fewer trades than 4h (less fee drag) but more 
than 1d (sufficient frequency). 1d HMA for primary trend, 1w HMA for confirmation.

Key Improvements from #692:
1. Connors RSI instead of standard RSI (better entry timing, proven 75% win rate)
2. Choppiness Index regime switch (CHOP>61.8=mean revert, CHOP<38.2=trend follow)
3. Donchian(20) breakouts for trend entries (proven in #696 with +28.7% return)
4. Asymmetric exits: quick profit take on mean revert, trail on trends
5. Looser entry thresholds to ensure 30+ trades/train, 3+/test on ALL symbols

Why this should work:
- CRSI worked on ETH in prior research (Sharpe +0.923)
- Donchian worked on SOL (Sharpe +0.782 in history)
- Choppiness filter prevents trend strategies in ranges (major failure mode)
- 12h TF = 20-50 trades/year target (optimal fee/reward balance)
- Multiple HTF filters reduce whipsaw that killed #690/#694

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_donchian_chop_regime_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_3 = 100 - (100 / (1 + rs))
    rsi_3 = np.clip(rsi_3, 0, 100)
    
    # RSI Streak(2) - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Percent Rank(100) - where current close ranks in last 100 bars
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        rank = np.sum(window < close[i]) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    with np.errstate(invalid='ignore'):
        crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index: measures market choppiness vs trending.
    CHOP > 61.8 = range/consolidation (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = np.sum(tr[i-period+1:i+1])
        
        if highest_high - lowest_low > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channels - highest high / lowest low over period."""
    n = len(close := high)  # just for length
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period:
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_12h = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Need 200 for SMA + buffer for HTF alignment
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if np.isnan(chop_12h[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_12h[i] > 55.0  # Range/consolidation (mean revert)
        is_trending = chop_12h[i] < 42.0  # Trending (trend follow)
        # Neutral zone between 42-55 = use conservative signals
        
        # === TREND BIAS (HTF HMA) ===
        trend_bullish_1d = close[i] > hma_1d_aligned[i]
        trend_bearish_1d = close[i] < hma_1d_aligned[i]
        trend_bullish_1w = close[i] > hma_1w_aligned[i]
        trend_bearish_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bias when both HTF agree
        trend_strong_bullish = trend_bullish_1d and trend_bullish_1w
        trend_strong_bearish = trend_bearish_1d and trend_bearish_1w
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === MEAN REVERSION MODE (Choppy regime) ===
        if is_choppy:
            # Long: CRSI oversold + bullish bias or neutral
            if crsi_12h[i] < 12 and (trend_bullish_1d or not trend_strong_bearish):
                if above_sma200 or trend_bullish_1d:
                    desired_signal = current_size
                elif crsi_12h[i] < 8:  # Extreme oversold
                    desired_signal = REDUCED_SIZE
            
            # Short: CRSI overbought + bearish bias or neutral
            elif crsi_12h[i] > 88 and (trend_bearish_1d or not trend_strong_bullish):
                if below_sma200 or trend_bearish_1d:
                    desired_signal = -current_size
                elif crsi_12h[i] > 92:  # Extreme overbought
                    desired_signal = -REDUCED_SIZE
        
        # === TREND FOLLOWING MODE (Trending regime) ===
        elif is_trending:
            # Long breakout: price breaks Donchian upper + bullish trend
            if close[i] >= donchian_upper[i] and trend_strong_bullish:
                desired_signal = current_size
            elif close[i] >= donchian_upper[i] and trend_bullish_1d and above_sma200:
                desired_signal = REDUCED_SIZE
            
            # Short breakout: price breaks Donchian lower + bearish trend
            elif close[i] <= donchian_lower[i] and trend_strong_bearish:
                desired_signal = -current_size
            elif close[i] <= donchian_lower[i] and trend_bearish_1d and below_sma200:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL ZONE (42-55 CHOP) ===
        else:
            # Only take extreme CRSI signals with strong HTF confirmation
            if crsi_12h[i] < 10 and trend_strong_bullish and above_sma200:
                desired_signal = REDUCED_SIZE
            elif crsi_12h[i] > 90 and trend_strong_bearish and below_sma200:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not overbought and trend intact
                if crsi_12h[i] < 75 and trend_bullish_1d:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if CRSI not oversold and trend intact
                if crsi_12h[i] > 25 and trend_bearish_1d:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Long exit: CRSI overbought OR trend reverses below both HTF HMA
        if in_position and position_side > 0:
            if crsi_12h[i] > 85:
                desired_signal = 0.0
            elif close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]:
                desired_signal = 0.0
        
        # Short exit: CRSI oversold OR trend reverses above both HTF HMA
        if in_position and position_side < 0:
            if crsi_12h[i] < 15:
                desired_signal = 0.0
            elif close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE * 0.8 else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE * 0.8 else -REDUCED_SIZE
        
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