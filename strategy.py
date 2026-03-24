#!/usr/bin/env python3
"""
Experiment #400: 6h Primary + 1d/1w HTF — Simplified Multi-TF Mean Reversion v1

Hypothesis: Previous 6h strategies failed due to overly complex regime detection
that rarely triggered (0 trades). This version uses SIMPLIFIED multi-TF logic:
- 1w HMA for major trend bias (slow, reduces whipsaw in bear markets)
- 1d RSI for intermediate momentum confirmation
- 6h RSI extremes for entry timing (mean reversion in range, trend in breakout)
- Choppiness Index as simple regime filter (not complex ADX+CHOP combo)

Key differences from failed #395, #398:
1. LOOSENED RSI thresholds (28/72 instead of strict CRSI bands)
2. Fewer confluence requirements (max 3 filters per entry)
3. 1w HMA instead of 1d for trend bias (slower = fewer false signals)
4. Simpler regime: CHOP > 55 = range (mean revert), CHOP < 45 = trend
5. Position sizing: 0.25 base, 0.30 when all HTF aligned

Entry Logic:
- Range Long: CHOP>55 + RSI(6h)<28 + price>1w_HMA + 1d_RSI<50
- Range Short: CHOP>55 + RSI(6h)>72 + price<1w_HMA + 1d_RSI>50
- Trend Long: CHOP<45 + RSI(6h)<35 + price>1w_HMA + 1d_RSI>50
- Trend Short: CHOP<45 + RSI(6h)>65 + price<1w_HMA + 1d_RSI<50

Target: 30-60 trades/year, Sharpe>0.40, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_simplified_rsi_hma_1d1w_v1"
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

def calculate_streak_rsi(close, period=2):
    """RSI of consecutive up/down days for Connors RSI"""
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    # Calculate streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_rsi = calculate_rsi(streak, period)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank for Connors RSI"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        returns = np.diff(close[i-period+1:i+1])
        current_return = returns[-1] if len(returns) > 0 else 0
        count_below = np.sum(returns[:-1] < current_return) if len(returns) > 1 else 0
        pr[i] = 100.0 * count_below / max(len(returns) - 1, 1)
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < pr_period + 1:
        return np.full(n, np.nan)
    
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_streak_rsi(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_short + streak_rsi + pr) / 3.0
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    rsi_1d_raw = calculate_rsi(df_1d['close'].values, period=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    rsi_6h = calculate_rsi(close, period=14)
    rsi_6h_short = calculate_rsi(close, period=6)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
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
        
        if np.isnan(rsi_6h[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (SIMPLIFIED) ===
        # CHOP > 55 = range (mean reversion)
        # CHOP < 45 = trend (trend following)
        # Otherwise = neutral (reduce size or flat)
        
        is_range = chop[i] > 55.0
        is_trend = chop[i] < 45.0
        
        # === HTF BIAS (1w HMA - major trend) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d MOMENTUM (RSI) ===
        htf_1d_bull = rsi_1d_aligned[i] > 50.0
        htf_1d_bear = rsi_1d_aligned[i] < 50.0
        
        # === 6h RSI EXTREMES (LOOSENED for more trades) ===
        rsi_oversold = rsi_6h[i] < 28.0
        rsi_overbought = rsi_6h[i] > 72.0
        rsi_oversold_mild = rsi_6h_short[i] < 35.0
        rsi_overbought_mild = rsi_6h_short[i] > 65.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC (SIMPLIFIED - fewer conditions) ===
        desired_signal = 0.0
        
        # REGIME 1: RANGE (mean reversion - primary strategy for bear/range markets)
        if is_range:
            # Long: RSI oversold + above 1w_HMA OR above SMA200
            if rsi_oversold and (htf_1w_bull or above_sma200):
                # Add 1d RSI confirmation (not too bullish = room to rebound)
                if rsi_1d_aligned[i] < 60.0:
                    desired_signal = SIZE_STRONG if htf_1w_bull else SIZE_BASE
            
            # Short: RSI overbought + below 1w_HMA OR below SMA200
            elif rsi_oversold == False and rsi_overbought and (htf_1w_bear or below_sma200):
                # Add 1d RSI confirmation (not too bearish = room to drop)
                if rsi_1d_aligned[i] > 40.0:
                    desired_signal = -SIZE_STRONG if htf_1w_bear else -SIZE_BASE
        
        # REGIME 2: TREND (trend following with pullback entries)
        elif is_trend:
            # Long: uptrend + mild RSI pullback
            if htf_1w_bull and htf_1d_bull and rsi_oversold_mild:
                desired_signal = SIZE_BASE
            
            # Short: downtrend + mild RSI bounce
            elif htf_1w_bear and htf_1d_bear and rsi_overbought_mild:
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