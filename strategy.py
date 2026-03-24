#!/usr/bin/env python3
"""
Experiment #452: 12h Primary + 1d HTF — Choppiness Regime + Connors RSI

Hypothesis: 12h timeframe captures multi-day swings without lower-TF noise.
Based on research showing Choppiness Index + Connors RSI achieved ETH Sharpe +0.923.

Key innovations vs failed #444 (12h CRSI + Choppiness, Sharpe=-2.925):
1. SIMPLER ENTRY: Max 2 conditions (not 4-5 confluence filters)
2. LOOSENED CRSI: Long <15 (not <10), Short >85 (not >90) — more trades
3. CHOP THRESHOLD: >55 (not >61.8) — more chop regimes qualify
4. NO DUAL HTF: Just 1d HMA bias (dual HTF was too restrictive in #444)
5. ATR STOP: 2.5x (not 2.0x) — less premature stops in volatile crypto

Entry Logic:
- Choppy Regime (CHOP>55): Connors RSI mean reversion
  Long: CRSI<15 + price>SMA50
  Short: CRSI>85 + price<SMA50
- Trending Regime (CHOP<45): HMA pullback entries
  Long: price>1d_HMA + pullback to 12h_HMA + RSI<50
  Short: price<1d_HMA + rally to 12h_HMA + RSI>50

Target: Sharpe>0.45, DD>-35%, trades>=40 train (10/year), trades>=6 test
Timeframe: 12h (proven higher TF works better for crypto)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_hma_1d_v2"
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
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    We use 55/45 thresholds for more regime switches
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR for each bar
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
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_pr=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of the streak length (consecutive up/down days)
    PercentRank: percentile rank of 1-day price change over 100 periods
    
    Long signal: CRSI < 10-15 (oversold)
    Short signal: CRSI > 85-90 (overbought)
    """
    n = len(close)
    if n < period_pr + 1:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi3 = calculate_rsi(close, period_rsi)
    
    # Streak calculation
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak (use absolute streak values for RSI calculation)
    streak_abs = np.abs(streak)
    streak_rsi = calculate_rsi(streak_abs, period_streak)
    
    # Percentile Rank of 1-day returns
    returns = np.zeros(n)
    returns[1:] = (close[1:] - close[:-1]) / close[:-1] * 100.0
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(period_pr, n):
        window = returns[i-period_pr+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            percent_rank[i] = np.sum(valid < returns[i]) / len(valid) * 100.0
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(period_pr, n):
        if not np.isnan(rsi3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

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
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_pr=100)
    sma_50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    SIZE_CHOP = 0.25  # Mean reversion in choppy regime
    SIZE_TREND = 0.30  # Trend following in trending regime
    
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
        
        if np.isnan(hma_12h[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_50[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = choppy/ranging (mean reversion)
        # CHOP < 45 = trending (trend following)
        # 45-55 = neutral (hold previous regime)
        
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === 1d HTF TREND BIAS ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h HMA POSITION ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === SMA50 FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === CRSI EXTREMES (LOOSENED: 15/85 instead of 10/90) ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === RSI PULLBACK FILTER ===
        rsi_pullback_long = rsi[i] < 50.0
        rsi_pullback_short = rsi[i] > 50.0
        
        # === ENTRY LOGIC (SIMPLE: max 2-3 conditions) ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (Connors RSI mean reversion)
        if is_choppy:
            # Long: CRSI oversold + above SMA50 (just 2 conditions!)
            if crsi_oversold and above_sma50:
                desired_signal = SIZE_CHOP
            
            # Short: CRSI overbought + below SMA50 (just 2 conditions!)
            elif crsi_overbought and below_sma50:
                desired_signal = -SIZE_CHOP
        
        # REGIME 2: TRENDING (HMA pullback with HTF bias)
        elif is_trending:
            # Long: 1d HTF bull + 12h HMA bull + RSI pullback
            if htf_bull and hma_bull and rsi_pullback_long:
                desired_signal = SIZE_TREND
            
            # Short: 1d HTF bear + 12h HMA bear + RSI pullback
            elif htf_bear and hma_bear and rsi_pullback_short:
                desired_signal = -SIZE_TREND
        
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
        if desired_signal >= SIZE_TREND * 0.9:
            final_signal = SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.9:
            final_signal = -SIZE_TREND
        elif desired_signal >= SIZE_CHOP * 0.9:
            final_signal = SIZE_CHOP
        elif desired_signal <= -SIZE_CHOP * 0.9:
            final_signal = -SIZE_CHOP
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