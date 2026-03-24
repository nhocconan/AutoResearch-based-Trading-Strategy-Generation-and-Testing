#!/usr/bin/env python3
"""
Experiment #422: 4h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime v1

Hypothesis: Recent failures show overly complex entry conditions = 0 trades.
Connors RSI (CRSI) has proven 75% win rate for mean reversion in bear/range markets.
Combined with Choppiness Index regime filter, this should work on ALL symbols.

Key innovations:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
2. Choppiness Index switches between trend-follow and mean-revert modes
3. 1d HMA for HTF trend bias (only trade with HTF direction)
4. Very simple entries: CRSI<15 long, CRSI>85 short + HTF alignment
5. ATR stoploss at 2.5x from entry

Why this should work:
- CRSI catches extreme oversold/overbought (works in 2022 crash & 2025 bear)
- CHOP filter avoids mean-revert in strong trends (reduces whipsaws)
- 1d HTF alignment ensures we trade with higher timeframe direction
- 4h TF = 20-50 trades/year target (fee-efficient)
- Simple conditions = enough trades on ALL symbols

Position sizing: 0.25 base, 0.30 when HTF strongly aligned
Stoploss: 2.5x ATR(14) from entry price
Target: Sharpe>0.45, DD>-35%, trades>=25 train, trades>=5 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_hma_1d1w_v1"
timeframe = "4h"
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

def calculate_rsi_streak(close, period=2):
    """RSI Streak component of Connors RSI
    Measures consecutive up/down days as RSI
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    # Calculate streak length
    streak = np.zeros(n)
    streak[:] = np.nan
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            if i > 0 and not np.isnan(streak[i-1]) and streak[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif close[i] < close[i-1]:
            if i > 0 and not np.isnan(streak[i-1]) and streak[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0 if np.isnan(streak[i-1]) else streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    # Positive streak = bullish, negative = bearish
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        if np.isnan(streak[i]):
            continue
        # Map streak to 0-100 scale
        # streak of +period = 100, -period = 0
        streak_rsi[i] = 50.0 + (streak[i] / period) * 50.0
        streak_rsi[i] = np.clip(streak_rsi[i], 0, 100)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank component of Connors RSI
    Where current price ranks in last N bars (0-100)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        # Count how many values in window are <= current
        count = np.sum(window <= current)
        pr[i] = (count / period) * 100.0
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < pr_period:
        return np.full(n, np.nan)
    
    rsi3 = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(pr_period, n):
        if np.isnan(rsi3[i]) or np.isnan(rsi_streak[i]) or np.isnan(percent_rank[i]):
            continue
        crsi[i] = (rsi3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
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
    """Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
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

def calculate_volume_sma(volume, period=20):
    """Volume SMA for confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    hma_4h = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, 200)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # CRSI thresholds (proven from literature)
    CRSI_LONG = 15.0   # Very oversold
    CRSI_SHORT = 85.0  # Very overbought
    
    # Choppiness thresholds
    CHOP_RANGE_HIGH = 61.8  # Mean revert mode
    CHOP_TREND_LOW = 38.2   # Trend follow mode
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(hma_4h[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = not np.isnan(chop[i]) and chop[i] > CHOP_RANGE_HIGH
        is_trending = not np.isnan(chop[i]) and chop[i] < CHOP_TREND_LOW
        
        # === CRSI SIGNALS ===
        crsi_oversold = crsi[i] < CRSI_LONG
        crsi_overbought = crsi[i] > CRSI_SHORT
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = False
        if not np.isnan(vol_sma[i]) and vol_sma[i] > 1e-10:
            vol_confirm = volume[i] > 1.2 * vol_sma[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (Mean Reversion with CRSI)
        if is_choppy:
            # Long: CRSI very oversold + above SMA200 + HTF not strongly bear
            if crsi_oversold and above_sma200:
                if htf_1d_bull or not htf_1d_bear:
                    desired_signal = SIZE_STRONG if vol_confirm else SIZE_BASE
            
            # Short: CRSI very overbought + below SMA200 + HTF not strongly bull
            elif crsi_overbought and below_sma200:
                if htf_1d_bear or not htf_1d_bull:
                    desired_signal = -SIZE_STRONG if vol_confirm else -SIZE_BASE
        
        # REGIME 2: TRENDING (Follow HTF direction with CRSI pullback)
        elif is_trending:
            # Long in uptrend: HTF bull + 4h bull + CRSI pullback (not extreme)
            if htf_1d_bull and hma_bull:
                if crsi[i] < 40.0:  # Pullback but not extreme
                    desired_signal = SIZE_BASE
            
            # Short in downtrend: HTF bear + 4h bear + CRSI bounce (not extreme)
            elif htf_1d_bear and hma_bear:
                if crsi[i] > 60.0:  # Bounce but not extreme
                    desired_signal = -SIZE_BASE
        
        # REGIME 3: NEUTRAL (Use CRSI extremes only)
        else:
            # Long: CRSI very oversold + above SMA200
            if crsi_oversold and above_sma200:
                desired_signal = SIZE_BASE
            
            # Short: CRSI very overbought + below SMA200
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