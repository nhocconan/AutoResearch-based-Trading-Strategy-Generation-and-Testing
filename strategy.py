#!/usr/bin/env python3
"""
Experiment #697: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: Daily timeframe with weekly trend filter provides optimal balance between
signal quality and trade frequency (target 25-40 trades/year). Choppiness Index detects
regime (range vs trend), Connors RSI provides high-probability mean-reversion entries
in ranges, and 1w HMA prevents counter-trend trades that destroyed capital in 2022.

Why this should work:
1. Connors RSI (CRSI) has 75% win rate in research - combines RSI(3) + Streak RSI + PercentRank
2. Choppiness Index > 61.8 = range market (mean revert), < 38.2 = trend (breakout)
3. 1w HMA filter prevents entering against major trend (critical for 2022 crash survival)
4. 1d TF worked for ETH in past experiments (Sharpe +0.923 with similar setup)
5. Looser CRSI thresholds (10/90 not 5/95) to ensure 30+ trades in train period

Key differences from failed strategies:
- Fewer filters than #693 (which had 0 trades) - only CHOP + CRSI + HMA
- CRSI instead of simple RSI - more reliable for mean reversion
- 1w HMA instead of 1d - stronger trend filter, fewer whipsaws
- ATR stoploss at 3x (not 2.5x) - gives trades more room in volatile crypto

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_hma_1w_v1"
timeframe = "1d"
leverage = 1.0

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
    Connors RSI - composite mean-reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Research shows 75% win rate for CRSI < 10 (long) and CRSI > 90 (short).
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak values
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Component 3: Percentile Rank of price change over rank_period
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0 and returns[-1] != 0:
            pct_rank[i] = np.sum(returns[:-1] <= returns[-1]) / len(returns[:-1]) * 100
        else:
            pct_rank[i] = 50.0
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    
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
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    crsi_1d = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
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
    
    for i in range(250, n):  # Need 200 for SMA + buffer for HTF alignment + CRSI rank_period
        # Skip if indicators not ready
        if np.isnan(crsi_1d[i]) or np.isnan(chop_1d[i]) or np.isnan(atr_1d[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            continue
        if atr_1d[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_1d[i] > 55.0  # Slightly lower threshold for more trades
        is_trending = chop_1d[i] < 45.0  # Slightly higher threshold for more trades
        # Neutral zone: 45-55 (no strong signal either way)
        
        # === TREND BIAS (1w HMA) ===
        trend_bullish = close[i] > hma_1w_aligned[i]
        trend_bearish = close[i] < hma_1w_aligned[i]
        
        # === SMA200 FILTER (long-term trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # Reduce size in high volatility (ATR expansion)
        atr_median = np.nanmedian(atr_1d[max(0,i-100):i+1])
        if atr_median > 0:
            atr_ratio = atr_1d[i] / (atr_median + 1e-10)
            if atr_ratio > 1.8:
                current_size = REDUCED_SIZE
        
        # === MEAN REVERSION MODE (Choppy/Range Market) ===
        if is_choppy:
            # Long: CRSI oversold + bullish or neutral HTF trend
            if crsi_1d[i] < 15 and (trend_bullish or not trend_bearish):
                desired_signal = current_size
            
            # Short: CRSI overbought + bearish or neutral HTF trend
            elif crsi_1d[i] > 85 and (trend_bearish or not trend_bullish):
                desired_signal = -current_size
            
            # Weaker signals with extreme CRSI
            elif crsi_1d[i] < 10:
                desired_signal = current_size * 0.7
            
            elif crsi_1d[i] > 90:
                desired_signal = -current_size * 0.7
        
        # === TREND FOLLOWING MODE (Trending Market) ===
        elif is_trending:
            # Long: pullback in uptrend (CRSI moderate low + bullish trend)
            if crsi_1d[i] < 40 and trend_bullish and above_sma200:
                desired_signal = current_size
            
            # Short: rally in downtrend (CRSI moderate high + bearish trend)
            elif crsi_1d[i] > 60 and trend_bearish and below_sma200:
                desired_signal = -current_size
        
        # === NEUTRAL ZONE (45-55 CHOP) ===
        else:
            # Only take extreme CRSI signals in neutral zone
            if crsi_1d[i] < 12 and trend_bullish:
                desired_signal = REDUCED_SIZE
            elif crsi_1d[i] > 88 and trend_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 3x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not overbought and trend still intact
                if crsi_1d[i] < 75 and trend_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if CRSI not oversold and trend still intact
                if crsi_1d[i] > 25 and trend_bearish:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Long exit: CRSI overbought OR trend reverses below 1w HMA
        if in_position and position_side > 0:
            if crsi_1d[i] > 85:
                desired_signal = 0.0
            elif close[i] < hma_1w_aligned[i] and chop_1d[i] < 40:
                desired_signal = 0.0
        
        # Short exit: CRSI oversold OR trend reverses above 1w HMA
        if in_position and position_side < 0:
            if crsi_1d[i] < 15:
                desired_signal = 0.0
            elif close[i] > hma_1w_aligned[i] and chop_1d[i] < 40:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            else:
                desired_signal = REDUCED_SIZE
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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