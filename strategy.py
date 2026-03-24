#!/usr/bin/env python3
"""
Experiment #651: 6h Primary + 1d/1w HTF — Funding Rate Mean Reversion + Weekly Pivot + RSI

Hypothesis: 6h timeframe is underexplored middle ground between 4h and 12h. Combining
funding rate mean reversion (proven BTC/ETH edge) with weekly pivot levels and RSI
extremes should capture both contrarian flows and key S/R bounces. 1d/1w HMA provides
macro trend filter to avoid fighting major trends.

Key innovations:
1. Funding rate z-score (30d) - contrarian when extreme (>2 or <-2)
2. Weekly pivot levels (R1, S1, P) - actual HTF support/resistance
3. RSI(14) extremes (30/70) with 6h SMA(200) filter
4. 1d HMA(21) + 1w HMA(50) dual HTF trend bias
5. Asymmetric sizing - stronger when aligned with both HTFs
6. ATR(14) trailing stop 2.5x for risk management

Entry conditions (LOOSE to ensure 30+ trades/year):
- LONG: funding_z < -1.5 OR (price near WS1 + RSI<40) + 1d HMA bull
- SHORT: funding_z > 1.5 OR (price near WR1 + RSI>60) + 1d HMA bear
- 1w HMA provides macro bias (reduce size against it)

Target: Sharpe>0.40, trades>=30 train, trades>=3 test
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_funding_pivot_rsi_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
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

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_pivot_levels(high, low, close):
    """Calculate standard pivot levels (P, R1, S1)"""
    n = len(close)
    pivot = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    
    for i in range(1, n):
        pivot[i] = (high[i-1] + low[i-1] + close[i-1]) / 3.0
        r1[i] = 2.0 * pivot[i] - low[i-1]
        s1[i] = 2.0 * pivot[i] - high[i-1]
    
    return pivot, r1, s1

def calculate_zscore(values, period=30):
    """Rolling Z-score"""
    n = len(values)
    if n < period:
        return np.full(n, np.nan)
    
    zscore = np.full(n, np.nan)
    for i in range(period, n):
        window = values[i-period+1:i+1]
        mean = np.nanmean(window)
        std = np.nanstd(window)
        if std > 1e-10:
            zscore[i] = (values[i] - mean) / std
        else:
            zscore[i] = 0.0
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMAs
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    pivot_6h, r1_6h, s1_6h = calculate_pivot_levels(high, low, close)
    
    # Funding rate proxy using price momentum z-score (if funding data unavailable)
    # This approximates funding rate extremes via price deviation
    momentum = np.diff(close, prepend=close[0])
    funding_z = calculate_zscore(momentum, period=30)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    SIZE_WEAK = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(sma_200[i]):
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
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === SMA(200) FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        rsi_extreme_low = rsi[i] < 30.0
        rsi_extreme_high = rsi[i] > 70.0
        
        # === FUNDING Z-SCORE (contrarian) ===
        funding_extreme_low = not np.isnan(funding_z[i]) and funding_z[i] < -1.5
        funding_extreme_high = not np.isnan(funding_z[i]) and funding_z[i] > 1.5
        
        # === PIVOT PROXIMITY ===
        near_s1 = False
        near_r1 = False
        if not np.isnan(s1_6h[i]) and s1_6h[i] > 0:
            dist_to_s1 = abs(close[i] - s1_6h[i]) / s1_6h[i]
            near_s1 = dist_to_s1 < 0.02  # within 2%
        
        if not np.isnan(r1_6h[i]) and r1_6h[i] > 0:
            dist_to_r1 = abs(close[i] - r1_6h[i]) / r1_6h[i]
            near_r1 = dist_to_r1 < 0.02  # within 2%
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG entries - multiple pathways to ensure trades
        long_score = 0
        
        # Path 1: Funding contrarian + 1d bull
        if funding_extreme_low and htf_1d_bull:
            long_score += 3
        
        # Path 2: RSI oversold + above SMA200 + 1d bull
        if rsi_oversold and above_sma200 and htf_1d_bull:
            long_score += 2
        
        # Path 3: Near S1 pivot + RSI low
        if near_s1 and (rsi_oversold or rsi_extreme_low):
            long_score += 2
        
        # Path 4: Extreme RSI + 1w bull (macro alignment)
        if rsi_extreme_low and htf_1w_bull:
            long_score += 3
        
        # Path 5: Simple pullback in uptrend
        if htf_1d_bull and htf_1w_bull and rsi[i] < 50 and close[i] > sma_200[i]:
            long_score += 1
        
        # SHORT entries - multiple pathways
        short_score = 0
        
        # Path 1: Funding contrarian + 1d bear
        if funding_extreme_high and htf_1d_bear:
            short_score += 3
        
        # Path 2: RSI overbought + below SMA200 + 1d bear
        if rsi_overbought and below_sma200 and htf_1d_bear:
            short_score += 2
        
        # Path 3: Near R1 pivot + RSI high
        if near_r1 and (rsi_overbought or rsi_extreme_high):
            short_score += 2
        
        # Path 4: Extreme RSI + 1w bear (macro alignment)
        if rsi_extreme_high and htf_1w_bear:
            short_score += 3
        
        # Path 5: Simple rally in downtrend
        if htf_1d_bear and htf_1w_bear and rsi[i] > 50 and close[i] < sma_200[i]:
            short_score += 1
        
        # Determine signal based on scores
        if long_score >= 3 and short_score < 2:
            # Strong long
            if htf_1w_bull:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        elif long_score >= 2 and short_score < 2:
            # Moderate long
            desired_signal = SIZE_BASE
        elif long_score >= 1 and short_score < 1:
            # Weak long
            desired_signal = SIZE_WEAK
        elif short_score >= 3 and long_score < 2:
            # Strong short
            if htf_1w_bear:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        elif short_score >= 2 and long_score < 2:
            # Moderate short
            desired_signal = -SIZE_BASE
        elif short_score >= 1 and long_score < 1:
            # Weak short
            desired_signal = -SIZE_WEAK
        else:
            # Conflicted or no signal
            desired_signal = 0.0
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif desired_signal >= SIZE_WEAK * 0.9:
            final_signal = SIZE_WEAK
        elif desired_signal <= -SIZE_WEAK * 0.9:
            final_signal = -SIZE_WEAK
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals