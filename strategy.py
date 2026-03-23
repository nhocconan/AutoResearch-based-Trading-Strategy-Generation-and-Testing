#!/usr/bin/env python3
"""
Experiment #097: 1d Primary + 1w HTF — Connors RSI Mean Reversion with Trend Bias

Hypothesis: Daily charts work best for mean reversion strategies. Connors RSI (CRSI)
has proven 75% win rate on daily timeframes. Combined with 1w HMA for macro trend
bias, this should generate 20-40 trades/year with positive Sharpe on ALL symbols.

Key innovations:
1) Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More responsive than standard RSI for daily entries
   - Proven on ETH (Sharpe +0.923 in research)
2) 1w HMA(21) for macro trend bias (not hard filter)
3) Choppiness Index as soft regime filter (boosts size, doesn't block trades)
4) ATR(14) trailing stop at 2.5x
5) Looser CRSI thresholds (15/85 instead of 10/90) to ensure trade generation
6) Discrete sizing: 0.25 base, 0.30 max with confluence

Why this should work on 1d:
- Daily timeframe naturally limits trades to 20-40/year
- CRSI catches extreme oversold/overbought conditions
- 1w HMA prevents counter-trend trades in 2022 crash and 2025 bear
- Softer filters ensure trades on BTC/ETH (not just SOL)
- Proven MTF structure from successful experiments

Position size: 0.25 base, 0.30 max with confluence
Stoploss: 2.5*ATR trailing
Target: 25-45 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_hma_trend_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_crsi(close):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days (streak length)
    PercentRank: percentile rank of daily return in last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - very short term
    rsi3 = calculate_rsi(close, period=3)
    
    # RSI Streak (2) - measure of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=2, min_periods=2, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=2, min_periods=2, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # PercentRank(100) - where current return ranks in last 100 days
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(100, n):
        window = returns[i-100:i]
        current = returns[i]
        rank = np.sum(window < current) / 100.0 * 100.0
        percent_rank[i] = rank
    percent_rank[:100] = 50.0
    
    # Combine into CRSI
    crsi = (rsi3 + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(sma_200[i]):
            continue
        
        # === HTF TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 50.0  # trending market
        chop_ranging = chop_14[i] > 50.0  # ranging market
        
        # === SMA FILTER ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CRSI ENTRY SIGNALS (looser thresholds for trade generation) ===
        crsi_oversold = crsi[i] < 20.0  # extreme oversold (was 10, loosened)
        crsi_overbought = crsi[i] > 80.0  # extreme overbought (was 90, loosened)
        crsi_neutral_long = crsi[i] < 40.0  # not overbought for long
        crsi_neutral_short = crsi[i] > 60.0  # not oversold for short
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: CRSI oversold + trend bias ---
        # Primary: CRSI < 20 (extreme oversold) + price > SMA200 (long-term uptrend)
        if crsi_oversold and price_above_sma200:
            new_signal = POSITION_SIZE_BASE
            # Boost if 1w HMA bullish + trending regime
            if price_above_hma_1w and chop_trending:
                new_signal = POSITION_SIZE_MAX
        # Secondary: CRSI < 40 + 1w HMA bullish (pullback in uptrend)
        elif crsi_neutral_long and price_above_hma_1w and price_above_sma200:
            new_signal = POSITION_SIZE_BASE
        
        # --- SHORT ENTRY: CRSI overbought + trend bias ---
        # Primary: CRSI > 80 (extreme overbought) + price < SMA200 (long-term downtrend)
        if crsi_overbought and price_below_sma200:
            new_signal = -POSITION_SIZE_BASE
            # Boost if 1w HMA bearish + trending regime
            if price_below_hma_1w and chop_trending:
                new_signal = -POSITION_SIZE_MAX
        # Secondary: CRSI > 60 + 1w HMA bearish (rally in downtrend)
        elif crsi_neutral_short and price_below_hma_1w and price_below_sma200:
            new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Keep position if CRSI hasn't reached extreme exit zone
        if in_position and new_signal == 0.0:
            if position_side > 0 and crsi[i] < 75.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and crsi[i] > 25.0:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE ===
        # Exit long if price drops below SMA200
        if in_position and position_side > 0:
            if price_below_sma200:
                new_signal = 0.0
        
        # Exit short if price rises above SMA200
        if in_position and position_side < 0:
            if price_above_sma200:
                new_signal = 0.0
        
        # === EXIT ON CRSI EXTREME (take profit) ===
        if in_position and position_side > 0 and crsi[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 25.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals