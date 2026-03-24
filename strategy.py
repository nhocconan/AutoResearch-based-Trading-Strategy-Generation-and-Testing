#!/usr/bin/env python3
"""
Experiment #382: 4h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime v1

Hypothesis: Previous strategies failed due to overly strict entry conditions (0 trades).
This version uses Connors RSI (proven 75% win rate) with simplified regime detection.

Key innovations:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More sensitive than standard RSI(14), catches reversals faster
   - Long when CRSI < 15, Short when CRSI > 85
2. Choppiness Index regime: CHOP > 61.8 = mean revert, CHOP < 38.2 = trend follow
3. 1d HMA for trend bias (only require alignment, not strict filter)
4. 1w HMA for major trend direction (avoid counter-trend trades)
5. VERY SIMPLE entries: CRSI extreme + regime match + HTF bias (3 filters max)
6. ATR(14) stoploss at 2.5x from entry

Position sizing: 0.25 base, 0.30 when 1d HTF aligned
Target: 30-50 trades/year, Sharpe > 0.5, DD < -30%

Why this should work:
- Connors RSI is proven for mean reversion (works in bear/range markets)
- Choppiness filter avoids trend-following in chop (where most strategies fail)
- 4h timeframe = 20-50 trades/year (optimal for fee/cost ratio)
- Simple conditions = actual trades generated (not 0 trades like #370, #373, #376)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_1d1w_v1"
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
    """RSI Streak Component for Connors RSI
    Counts consecutive up/down days, applies RSI to streak values
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    # Calculate streak values
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Apply RSI to streak values
    rsi_streak = calculate_rsi(streak, period)
    return rsi_streak

def calculate_percent_rank(close, period=100):
    """Percent Rank for Connors RSI
    Percentage of prior returns less than current return
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate returns
    returns = np.zeros(n)
    returns[:] = np.nan
    for i in range(1, n):
        if close[i-1] > 1e-10:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
    
    # Calculate percent rank
    pr = np.zeros(n)
    pr[:] = np.nan
    for i in range(period, n):
        count_below = np.sum(returns[i-period:i] < returns[i])
        pr[i] = 100.0 * count_below / period
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < pr_period + 5:
        return np.full(n, np.nan)
    
    rsi_short = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    hma_4h = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, 200)
    
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
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(hma_4h[i]):
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
        
        # === REGIME DETECTION with Choppiness ===
        # CHOP > 61.8 = choppy/range (mean reversion)
        # CHOP < 38.2 = trending (trend follow)
        # Between = neutral (use previous or reduce size)
        
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === HTF BIAS (1d and 1w) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === SMA200 FILTER ===
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === CONNORS RSI EXTREMES ===
        # CRSI < 15 = oversold (long opportunity)
        # CRSI > 85 = overbought (short opportunity)
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === ENTRY LOGIC (SIMPLIFIED - 3 filters max) ===
        desired_signal = 0.0
        
        # REGIME: CHOPPY (mean reversion - primary strategy)
        if is_choppy:
            # Long: CRSI oversold + 1d HTF bull (2 conditions)
            if crsi_oversold and htf_1d_bull:
                desired_signal = SIZE_STRONG if htf_1w_bull else SIZE_BASE
            
            # Short: CRSI overbought + 1d HTF bear (2 conditions)
            elif crsi_overbought and htf_1d_bear:
                desired_signal = -SIZE_STRONG if htf_1w_bear else -SIZE_BASE
        
        # REGIME: TRENDING (trend follow - secondary)
        elif is_trending:
            # Long: CRSI pullback + all HTF bull
            if crsi_oversold and htf_1d_bull and htf_1w_bull and hma_bull:
                desired_signal = SIZE_BASE
            
            # Short: CRSI rally + all HTF bear
            elif crsi_overbought and htf_1d_bear and htf_1w_bear and hma_bear:
                desired_signal = -SIZE_BASE
        
        # REGIME: NEUTRAL (reduced size mean reversion)
        else:
            # Only take high-conviction mean reversion
            if crsi_oversold and htf_1d_bull and above_sma200:
                desired_signal = SIZE_BASE * 0.6
            elif crsi_overbought and htf_1d_bear and below_sma200:
                desired_signal = -SIZE_BASE * 0.6
        
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.5
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.5
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