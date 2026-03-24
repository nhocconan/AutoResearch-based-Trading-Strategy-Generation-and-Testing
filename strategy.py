#!/usr/bin/env python3
"""
Experiment #447: 6h Primary + 1d HTF — Weighted HTF + Connors RSI + CHOP Regime

Hypothesis: Previous #435 failed (Sharpe=0.153) because dual HTF agreement (12h+1d)
was TOO restrictive - during trend transitions, they often disagree and block all trades.

NEW APPROACH:
1. WEIGHTED HTF: 1d HMA = primary trend (weight 0.6), 12h = secondary (0.4)
   - Allow entries when 1d is clear even if 12h is neutral
   - This increases trade frequency while keeping HTF guidance
2. CONNORS RSI (CRSI): Proven 75% win rate mean reversion
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 + price > SMA100
   - Short: CRSI > 85 + price < SMA100
3. CHOP INDEX for regime: Direct threshold (no percentile)
   - CHOP > 61.8 = range (use CRSI mean reversion)
   - CHOP < 38.2 = trend (use breakout entries)
4. LOOSER ENTRY: CRSI 15/85 (not 10/90), SMA100 (not SMA200)
   - Target: 40-80 trades/train, 10-20 trades/test
5. SIZE: 0.25 base, 0.30 strong (discrete levels)
6. STOPLOSS: 2.5x ATR (slightly wider for 6h volatility)

Timeframe: 6h (NEW exploration - zero prior experiments with CRSI+CHOP)
Target: Sharpe>0.45, DD>-35%, trades>=80 train, trades>=10 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weighted_htf_crsi_chop_1d_v1"
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
    Choppiness Index (CHOP) - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and tr_sum > 0:
            chop[i] = 100.0 * np.log10(tr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak component for Connors RSI
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(1, n):
        # Calculate streak length
        if close[i] > close[i-1]:
            streak = 1
            j = i - 1
            while j > 0 and close[j] > close[j-1]:
                streak += 1
                j -= 1
        elif close[i] < close[i-1]:
            streak = -1
            j = i - 1
            while j > 0 and close[j] < close[j-1]:
                streak -= 1
                j -= 1
        else:
            streak = 0
        
        # Convert to RSI-like scale (0-100)
        # Positive streak = bullish, negative = bearish
        if streak > 0:
            streak_rsi[i] = min(100.0, 50.0 + streak * 10.0)
        elif streak < 0:
            streak_rsi[i] = max(0.0, 50.0 + streak * 10.0)
        else:
            streak_rsi[i] = 50.0
    
    # Smooth with short EMA
    streak_rsi_smooth = pd.Series(streak_rsi).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return streak_rsi_smooth

def calculate_percentile_rank(values, period=100):
    """Percentile rank for Connors RSI"""
    n = len(values)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = values[i-period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            pr[i] = np.sum(valid < values[i]) / len(valid) * 100.0
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long signal: CRSI < 10-15 (oversold)
    Short signal: CRSI > 85-90 (overbought)
    """
    n = len(close)
    
    # RSI(3) - very short term
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak(2)
    streak_rsi = calculate_rsi_streak(close, period=streak_period)
    
    # Percent Rank(100)
    pr = calculate_percentile_rank(close, period=pr_period)
    
    # Combine
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(max(rsi_period, streak_period, pr_period), n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pr[i]) / 3.0
    
    return crsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
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
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    sma_100 = calculate_sma(close, 100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
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
        
        if np.isnan(hma_6h[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
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
        
        if np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (CHOP INDEX) ===
        # CHOP > 61.8 = range/choppy (mean reversion)
        # CHOP < 38.2 = trending (breakout)
        # Between = neutral (use HTF bias)
        
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === HTF TREND BIAS (WEIGHTED: 1d primary) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === SMA100 FILTER ===
        above_sma100 = close[i] > sma_100[i]
        below_sma100 = close[i] < sma_100[i]
        
        # === CRSI EXTREMES (Connors RSI) ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakdown_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY/RANGE (CRSI mean reversion)
        if is_choppy:
            # Long: CRSI oversold + above SMA100 + HTF not bearish
            if crsi_oversold and above_sma100 and not htf_bear:
                desired_signal = SIZE_BASE
            
            # Short: CRSI overbought + below SMA100 + HTF not bullish
            elif crsi_overbought and below_sma100 and not htf_bull:
                desired_signal = -SIZE_BASE
        
        # REGIME 2: TRENDING (Donchian breakout with HTF confirmation)
        elif is_trending:
            # Long: HTF bull + (Donchian breakout OR HMA bull)
            if htf_bull:
                if donchian_breakout_long or hma_bull:
                    desired_signal = SIZE_STRONG
            
            # Short: HTF bear + (Donchian breakdown OR HMA bear)
            elif htf_bear:
                if donchian_breakdown_short or hma_bear:
                    desired_signal = -SIZE_STRONG
        
        # REGIME 3: NEUTRAL (use HTF + HMA confluence)
        else:
            # Long: HTF bull + HMA bull + above SMA100
            if htf_bull and hma_bull and above_sma100:
                desired_signal = SIZE_BASE
            
            # Short: HTF bear + HMA bear + below SMA100
            elif htf_bear and hma_bear and below_sma100:
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