#!/usr/bin/env python3
"""
Experiment #444: 12h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime + Dual HTF

Hypothesis: 12h timeframe is ideal for swing trading - captures multi-day moves without
the noise of lower TFs. Recent failures show:
- Simple HMA/EMA crossover fails on BTC/ETH (whipsaw in 2022)
- Fisher transform strategies failing (#441-443)
- Weekly pivot filters too restrictive (0 trades)

New approach based on proven research:
1. CONNORS RSI (CRSI): 3-component RSI for mean reversion (75% win rate documented)
   - RSI(3) + RSI_Streak(2) + PercentRank(100) / 3
   - Long: CRSI < 10, Short: CRSI > 90
2. CHOPPINESS INDEX regime filter: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend
   - Best meta-filter for bear/range markets (2022, 2025)
3. DUAL HTF BIAS: 1d AND 1w HMA must agree for trend entries
4. ASYMMETRIC LOGIC: Mean revert in chop, trend-follow only when CHOP < 38.2
5. VOL-ADJUSTED SIZE: Reduce position size when ATR ratio spikes

Entry Logic:
- Mean Revert Long: CHOP > 61.8 + CRSI < 15 + price > 1d HMA
- Mean Revert Short: CHOP > 61.8 + CRSI > 85 + price < 1d HMA
- Trend Long: CHOP < 38.2 + 1d HMA bull + 1w HMA bull + HMA cross
- Trend Short: CHOP < 38.2 + 1d HMA bear + 1w HMA bear + HMA cross

Target: Sharpe>0.45, DD>-35%, trades>=40 train (10/year), trades>=5 test
Timeframe: 12h (proven higher TF works best for BTC/ETH)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_dual_htf_v1"
timeframe = "12h"
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
    """RSI of streak - consecutive up/down days"""
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    # Calculate streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = max(1, streak[i-1] + 1) if i > 0 and streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = min(-1, streak[i-1] - 1) if i > 0 and streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert to 0-100 scale (like RSI)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(period, n):
        positive_streaks = sum(1 for j in range(i-period+1, i+1) if streak[j] > 0)
        streak_rsi[i] = 100.0 * positive_streaks / period
    
    return streak_rsi

def calculate_percentile_rank(values, period=100):
    """Percentile rank of current value vs last N periods"""
    n = len(values)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = values[i-period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            pr[i] = np.sum(valid < values[i]) / len(valid) * 100.0
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) - 3-component mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Entry signals:
    - CRSI < 10: oversold (long)
    - CRSI > 90: overbought (short)
    """
    n = len(close)
    if n < pr_period + 5:
        return np.full(n, np.nan)
    
    rsi_short = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pr = calculate_percentile_rank(close, pr_period)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + pr[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8: choppy/range (mean reversion works)
    - CHOP < 38.2: trending (trend following works)
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
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        hl_range = highest_high - lowest_low
        
        if hl_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / hl_range) / np.log10(period)
    
    return chop

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

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR ratio for vol spike detection"""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    ratio = np.zeros(len(close))
    ratio[:] = np.nan
    for i in range(len(close)):
        if not np.isnan(atr_short[i]) and not np.isnan(atr_long[i]) and atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

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
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    hma_12h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, 7, 30)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    sma_100 = calculate_sma(close, 100)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_REDUCED = 0.15  # For high vol regimes
    
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
        
        if np.isnan(hma_12h[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_100[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        choppy_regime = chop[i] > 61.8  # Mean reversion works
        trending_regime = chop[i] < 38.2  # Trend following works
        neutral_regime = not choppy_regime and not trending_regime
        
        # === VOL REGIME (ATR ratio) ===
        vol_spike = not np.isnan(atr_ratio[i]) and atr_ratio[i] > 2.0
        current_size = SIZE_REDUCED if vol_spike else SIZE_BASE
        
        # === DUAL HTF BIAS (1d + 1w must agree for trend) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        htf_both_bull = htf_1d_bull and htf_1w_bull
        htf_both_bear = htf_1d_bear and htf_1w_bear
        
        # === 12h HMA TREND ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_12h_fast[i]) and not np.isnan(hma_12h_fast[i-1]):
            if not np.isnan(hma_12h[i]) and not np.isnan(hma_12h[i-1]):
                if hma_12h_fast[i-1] <= hma_12h[i-1] and hma_12h_fast[i] > hma_12h[i]:
                    hma_cross_long = True
                if hma_12h_fast[i-1] >= hma_12h[i-1] and hma_12h_fast[i] < hma_12h[i]:
                    hma_cross_short = True
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Very oversold
        crsi_overbought = crsi[i] > 85.0  # Very overbought
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === SMA100 FILTER ===
        above_sma100 = close[i] > sma_100[i]
        below_sma100 = close[i] < sma_100[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (Mean Reversion with CRSI)
        if choppy_regime:
            # Long: CRSI extreme oversold + above SMA100 (bullish bias)
            if crsi_extreme_oversold and above_sma100:
                desired_signal = current_size
            # Short: CRSI extreme overbought + below SMA100 (bearish bias)
            elif crsi_extreme_overbought and below_sma100:
                desired_signal = -current_size
            # Less extreme: require HTF alignment
            elif crsi_oversold and htf_1d_bull:
                desired_signal = current_size * 0.8
            elif crsi_overbought and htf_1d_bear:
                desired_signal = -current_size * 0.8
        
        # REGIME 2: TRENDING (Trend Following with HTF alignment)
        elif trending_regime:
            # Long: Dual HTF bull + HMA bull + (crossover or pullback)
            if htf_both_bull and hma_bull:
                if hma_cross_long or crsi_oversold:
                    desired_signal = SIZE_STRONG
            # Short: Dual HTF bear + HMA bear + (crossover or pullback)
            elif htf_both_bear and hma_bear:
                if hma_cross_short or crsi_overbought:
                    desired_signal = -SIZE_STRONG
        
        # REGIME 3: NEUTRAL (Conservative - only extreme CRSI)
        else:
            if crsi_extreme_oversold and htf_1d_bull:
                desired_signal = SIZE_REDUCED
            elif crsi_extreme_overbought and htf_1d_bear:
                desired_signal = -SIZE_REDUCED
        
        # === STOPLOSS CHECK (2.5x ATR from entry for 12h TF) ===
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
        elif desired_signal >= SIZE_REDUCED * 0.9:
            final_signal = SIZE_REDUCED
        elif desired_signal <= -SIZE_REDUCED * 0.9:
            final_signal = -SIZE_REDUCED
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
                # Set stoploss (2.5x ATR for 12h timeframe)
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