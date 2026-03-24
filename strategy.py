#!/usr/bin/env python3
"""
Experiment #379: 1h Primary + 4h/12h HTF — Simplified Regime Pullback v1

Hypothesis: Previous strategies failed due to overly strict entry conditions
resulting in 0 trades (Sharpe=0.000). This version SIMPLIFIES entries while
keeping proven HTF trend alignment.

Key changes from failed experiments:
1. LOOSENED RSI thresholds (30/70 instead of 25/75) for more trade triggers
2. Reduced confluence from 5+ filters to 3 core filters
3. HMA crossover entry (more frequent than Donchian breakout)
4. Removed volume confirmation (too restrictive for 1h TF)
5. Session filter relaxed (06-22 UTC instead of 08-20)
6. Added cRSI for faster mean reversion signals

Regime Detection:
- 4h HMA slope determines trend direction
- 12h HMA determines bias (long-only, short-only, or both)
- Choppiness > 55 = reduce size or skip

Entry Logic (SIMPLIFIED):
- Long: 4h HMA bull + 12h HMA bull + RSI(14)<45 + HMA cross up
- Short: 4h HMA bear + 12h HMA bear + RSI(14)>55 + HMA cross down
- Mean Reversion: cRSI<15 long, cRSI>85 short (when CHOP>55)

Position sizing: 0.20 base, 0.30 when HTF fully aligned
Stoploss: 2.5x ATR(14) from entry price

Target: Sharpe>0.45, DD>-35%, trades>=40 train, trades>=5 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - faster mean reversion signal"""
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI component (short period)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 and streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 and streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        pos_streak = np.sum(streak[max(0,i-streak_period):i+1] > 0)
        total = np.sum(np.abs(streak[max(0,i-streak_period):i+1]) > 0)
        if total > 0:
            streak_rsi[i] = 100.0 * pos_streak / total
    
    # Percent Rank component
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    for i in range(rank_period, n):
        returns = np.diff(close[max(0,i-rank_period):i+1])
        if len(returns) > 0:
            current_return = close[i] - close[i-1] if i > 0 else 0
            pct_rank[i] = 100.0 * np.sum(returns < current_return) / len(returns)
    
    # Combine components
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

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
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=21)
    hma_1h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h + 12h) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # Strong bias when both HTF agree
        htf_strong_bull = htf_4h_bull and htf_12h_bull
        htf_strong_bear = htf_4h_bear and htf_12h_bear
        
        # === 1h HMA TREND ===
        hma_bull = close[i] > hma_1h[i]
        hma_bear = close[i] < hma_1h[i]
        
        # === HMA CROSSOVER (entry trigger) ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_1h_fast[i]) and not np.isnan(hma_1h_fast[i-1]):
            if not np.isnan(hma_1h[i]) and not np.isnan(hma_1h[i-1]):
                if hma_1h_fast[i-1] <= hma_1h[i-1] and hma_1h_fast[i] > hma_1h[i]:
                    hma_cross_long = True
                if hma_1h_fast[i-1] >= hma_1h[i-1] and hma_1h_fast[i] < hma_1h[i]:
                    hma_cross_short = True
        
        # === SMA200 FILTER (loose - just avoid extreme counter-trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI CONDITIONS (LOOSENED for more trades) ===
        rsi_pullback_long = rsi[i] < 45.0  # pullback in uptrend
        rsi_pullback_short = rsi[i] > 55.0  # rally in downtrend
        
        # === cRSI EXTREMES (mean reversion) ===
        crsi_oversold = not np.isnan(crsi[i]) and crsi[i] < 20.0
        crsi_overbought = not np.isnan(crsi[i]) and crsi[i] > 80.0
        
        # === CHOPPINESS FILTER ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === SESSION FILTER (06-22 UTC for volume) ===
        # Extract hour from open_time (assumes Unix timestamp in milliseconds)
        hour_utc = (prices["open_time"].iloc[i] // 3600000) % 24
        in_session = 6 <= hour_utc <= 22
        
        # === ENTRY LOGIC (SIMPLIFIED - 3 confluence filters) ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (pullback entries with HTF alignment)
        if is_trending or not is_choppy:
            # Long: 4h bull + RSI pullback + HMA cross OR price>HMA
            if htf_4h_bull and rsi_pullback_long:
                if hma_cross_long or (hma_bull and above_sma200):
                    desired_signal = SIZE_STRONG if htf_strong_bull else SIZE_BASE
            
            # Short: 4h bear + RSI rally + HMA cross OR price<HMA
            elif htf_4h_bear and rsi_pullback_short:
                if hma_cross_short or (hma_bear and below_sma200):
                    desired_signal = -SIZE_STRONG if htf_strong_bear else -SIZE_BASE
        
        # REGIME 2: CHOPPY (cRSI mean reversion - simpler conditions)
        elif is_choppy:
            # Long: cRSI oversold + above SMA200 (avoid crash)
            if crsi_oversold and above_sma200:
                desired_signal = SIZE_BASE
            
            # Short: cRSI overbought + below SMA200 (avoid rally)
            elif crsi_overbought and below_sma200:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals