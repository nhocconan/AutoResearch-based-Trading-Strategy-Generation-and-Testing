#!/usr/bin/env python3
"""
Experiment #712: 12h Primary + 1d/1w HTF — Choppiness Regime + Connors RSI + Donchian

Hypothesis: 12h timeframe balances trade frequency (20-50/year) with signal quality.
Use Choppiness Index to detect regime, then apply appropriate strategy:
- CHOP > 55 (Range): Connors RSI mean reversion at extremes
- CHOP < 45 (Trend): Donchian breakout with 1d HMA trend filter

Why this should work:
1. Choppiness Index proven on ETH (Sharpe +0.923 in research)
2. Connors RSI has 75% win rate in mean reversion
3. 12h TF worked in research (SOL Sharpe +0.782 with Donchian+HMA)
4. Dual regime adapts to market conditions (unlike pure trend that failed #685/#690)
5. Simpler than failed CRSI+Chop combos (#701, #702, #707, #708)
6. Looser thresholds ensure trade frequency (avoid 0-trade failures)

Key differences from failed experiments:
- No volume filter (was causing 0 trades in #704, #708, #710)
- Simpler regime detection (CHOP only, not CHOP+ADX+CRSI combo)
- Connors RSI instead of regular RSI (better for mean reversion)
- 12h primary instead of 4h/1h (fewer trades, less fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_crsi_donchian_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
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
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion at extremes (<10 long, >90 short)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3) - very short term for quick reversals
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI - measures consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI of streak
    streak_abs = np.abs(streak)
    streak_gain = np.where(streak > 0, streak_abs, 0)
    streak_loss = np.where(streak < 0, streak_abs, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rsi = 100 - (100 / (1 + avg_streak_gain / (avg_streak_loss + 1e-10)))
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank - where current return ranks vs last 100 periods
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.concatenate([[0], returns])
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current) / len(window) * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures if market is trending or ranging.
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period * 2:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High - Lowest Low over period
    hh_ll = pd.Series(high).rolling(window=period, min_periods=period).max().values - \
            pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100 * np.log10(atr_sum / (hh_ll + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop_raw, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    chop_12h = calculate_choppiness(high, low, close, period=14)
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
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
        if np.isnan(chop_12h[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or atr_12h[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_range = chop_12h[i] > 55  # Ranging market
        regime_trend = chop_12h[i] < 45  # Trending market
        # Neutral zone: 45-55 (wait for clearer signal)
        
        # === TREND BIAS (1d and 1w HTF HMA) ===
        trend_bullish_1d = close[i] > hma_1d_aligned[i]
        trend_bearish_1d = close[i] < hma_1d_aligned[i]
        trend_bullish_1w = close[i] > hma_1w_aligned[i]
        trend_bearish_1w = close[i] < hma_1w_aligned[i]
        
        # Strong trend bias when both 1d and 1w agree
        trend_bullish_strong = trend_bullish_1d and trend_bullish_1w
        trend_bearish_strong = trend_bearish_1d and trend_bearish_1w
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === RANGE REGIME (CHOP > 55) - Mean Reversion with CRSI ===
        if regime_range:
            # Long: CRSI extremely oversold + above SMA200 (bullish bias)
            if crsi_12h[i] < 15 and above_sma200:
                desired_signal = current_size
            # Long: CRSI very oversold (weaker signal)
            elif crsi_12h[i] < 10:
                desired_signal = REDUCED_SIZE
            # Short: CRSI extremely overbought + below SMA200 (bearish bias)
            elif crsi_12h[i] > 85 and below_sma200:
                desired_signal = -current_size
            # Short: CRSI very overbought (weaker signal)
            elif crsi_12h[i] > 90:
                desired_signal = -REDUCED_SIZE
        
        # === TREND REGIME (CHOP < 45) - Donchian Breakout ===
        elif regime_trend:
            # Long breakout: price breaks Donchian upper + bullish trend bias
            if close[i] > donchian_upper[i-1] and trend_bullish_strong:
                desired_signal = current_size
            # Long breakout: price breaks Donchian upper + 1d bullish
            elif close[i] > donchian_upper[i-1] and trend_bullish_1d:
                desired_signal = REDUCED_SIZE
            # Short breakout: price breaks Donchian lower + bearish trend bias
            elif close[i] < donchian_lower[i-1] and trend_bearish_strong:
                desired_signal = -current_size
            # Short breakout: price breaks Donchian lower + 1d bearish
            elif close[i] < donchian_lower[i-1] and trend_bearish_1d:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Only take strongest CRSI signals in neutral
            if crsi_12h[i] < 8 and trend_bullish_1d:
                desired_signal = REDUCED_SIZE
            elif crsi_12h[i] > 92 and trend_bearish_1d:
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
        if in_position and position_side > 0:
            # Exit long on CRSI overbought or trend reversal
            if crsi_12h[i] > 80:
                desired_signal = 0.0
            elif close[i] < hma_1d_aligned[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short on CRSI oversold or trend reversal
            if crsi_12h[i] < 20:
                desired_signal = 0.0
            elif close[i] > hma_1d_aligned[i]:
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