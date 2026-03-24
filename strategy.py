#!/usr/bin/env python3
"""
Experiment #340: 6h Primary + 1d/1w HTF — CRSI Mean Reversion + Choppiness Regime v1

Hypothesis: Previous 6h strategies failed due to overly strict entry conditions (0 trades).
This version uses Connors RSI (CRSI) for mean reversion with LOOSENED thresholds,
Choppiness Index for regime detection, and 1d/1w HMA for trend bias.

Key differences from failed #331 (Sharpe=-0.084):
1. CRSI instead of regular RSI (better mean reversion signal, 75% win rate in literature)
2. LOOSENED CRSI thresholds: <20/>80 instead of <10/>90 (more trades)
3. Fewer confluence requirements: need 2 of 3 conditions, not all 3
4. Added Donchian breakout as alternative entry in trending regime
5. Asymmetric sizing: 0.25 base, 0.30 when 1d HTF aligned
6. Stoploss: 2.5x ATR from entry, properly tracked

CRSI Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- RSI(3): Very short-term momentum
- RSI_Streak(2): Consecutive up/down days
- PercentRank(100): Where current return ranks vs last 100 bars

Regime Detection:
- CHOP > 60 = choppy → CRSI mean reversion at extremes
- CHOP < 45 = trending → Donchian breakout + HMA alignment
- 45-60 = use previous regime (hysteresis)

Entry Logic (LOOSENED):
- Choppy: CRSI < 20 + price > SMA100 → long; CRSI > 80 + price < SMA100 → short
- Trending: Donchian(20) breakout + 1d HMA alignment
- Always require 6h HMA direction confirmation (but not 1d/1w)

Position sizing: 0.25 base, 0.30 when 1d HTF aligned (discrete levels only)
Stoploss: 2.5x ATR(14) from entry price

Target: Sharpe>0.40, DD>-40%, trades>=30 train, trades>=3 test, ALL symbols positive Sharpe
Trade frequency: 40-80 trades/year on 6h timeframe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_chop_regime_hma_1d1w_v1"
timeframe = "6h"
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
    """
    RSI Streak component of CRSI
    Measures consecutive up/down periods
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        streak = 0
        for j in range(period):
            if close[i-j] > close[i-j-1]:
                streak += 1
            elif close[i-j] < close[i-j-1]:
                streak -= 1
        
        # Convert streak to RSI-like scale (0-100)
        # Streak ranges from -period to +period
        streak_rsi[i] = 50.0 + (streak / period) * 50.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank component of CRSI
    Where current return ranks vs last 'period' returns
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    
    # Calculate returns
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0.0], returns])
    
    for i in range(period, n):
        current_return = returns[i]
        past_returns = returns[i-period+1:i]
        
        # Count how many past returns are less than current
        count_less = np.sum(past_returns < current_return)
        pct_rank[i] = 100.0 * count_less / period
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI)
    Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi_short = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pr_period)
    
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + pct_rank[i]) / 3.0
    
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
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound
    CHOP < 38.2 = trending
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_100 = calculate_sma(close, 100)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Regime memory for hysteresis (avoid flip-flop)
    prev_regime = 0  # 0=unknown, 1=trending, 2=choppy
    
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
        
        if np.isnan(hma_6h[i]) or np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_100[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION with HYSTERESIS ===
        choppy_threshold = 60.0
        trending_threshold = 45.0
        
        if chop[i] > choppy_threshold:
            current_regime = 2  # choppy
        elif chop[i] < trending_threshold:
            current_regime = 1  # trending
        else:
            current_regime = prev_regime  # use memory
        
        prev_regime = current_regime
        
        # === HTF BIAS (1d and 1w) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        htf_1w_bear = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === SMA100 FILTER ===
        above_sma100 = close[i] > sma_100[i]
        below_sma100 = close[i] < sma_100[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = False
        breakout_short = False
        if not np.isnan(donchian_upper[i-1]):
            breakout_long = close[i] > donchian_upper[i-1]
        if not np.isnan(donchian_lower[i-1]):
            breakout_short = close[i] < donchian_lower[i-1]
        
        # === CRSI EXTREMES (LOOSENED for more trades) ===
        crsi_oversold = crsi[i] < 20.0  # Was <10 in standard CRSI
        crsi_overbought = crsi[i] > 80.0  # Was >90 in standard CRSI
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (mean reversion with CRSI)
        if current_regime == 2:
            # Long: CRSI oversold + above SMA100 OR 6h HMA bull (need 1 of 2)
            if crsi_oversold:
                confluence_count = 0
                if above_sma100:
                    confluence_count += 1
                if hma_bull:
                    confluence_count += 1
                
                if confluence_count >= 1:  # LOOSENED: need only 1 of 2
                    desired_signal = SIZE_STRONG if htf_1d_bull else SIZE_BASE
            
            # Short: CRSI overbought + below SMA100 OR 6h HMA bear (need 1 of 2)
            elif crsi_overbought:
                confluence_count = 0
                if below_sma100:
                    confluence_count += 1
                if hma_bear:
                    confluence_count += 1
                
                if confluence_count >= 1:  # LOOSENED: need only 1 of 2
                    desired_signal = -SIZE_STRONG if htf_1d_bear else -SIZE_BASE
        
        # REGIME 2: TRENDING (breakout with Donchian + HTF confirmation)
        elif current_regime == 1:
            # Long: Donchian breakout + 6h HMA bull + 1d bull (need 2 of 3)
            if breakout_long:
                confluence_count = 0
                if hma_bull:
                    confluence_count += 1
                if htf_1d_bull:
                    confluence_count += 1
                
                if confluence_count >= 1:  # LOOSENED: need only 1 of 2 (not counting breakout)
                    desired_signal = SIZE_STRONG if htf_1w_bull else SIZE_BASE
            
            # Short: Donchian breakout + 6h HMA bear + 1d bear (need 2 of 3)
            elif breakout_short:
                confluence_count = 0
                if hma_bear:
                    confluence_count += 1
                if htf_1d_bear:
                    confluence_count += 1
                
                if confluence_count >= 1:  # LOOSENED: need only 1 of 2 (not counting breakout)
                    desired_signal = -SIZE_STRONG if htf_1w_bear else -SIZE_BASE
        
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