#!/usr/bin/env python3
"""
Experiment #442: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + Fisher Reversals

Hypothesis: 4h timeframe proven to work well (20-50 trades/year target).
Combining KAMA (adaptive to volatility) with Ehlers Fisher Transform (reversal detector)
should capture both trending and mean-reverting phases better than pure HMA.

Key innovations:
1. KAMA instead of HMA/EMA - adapts to market efficiency ratio (ER)
2. Fisher Transform for reversal entries (proven in bear markets)
3. Choppiness Index regime filter (CHOP>61.8=range, CHOP<38.2=trend)
4. Dual HTF bias: 1d AND 1w KAMA must agree for trend entries
5. LOOSE entry conditions: Fisher crosses -1.5/+1.5 OR KAMA cross OR RSI extremes

Why this should work:
- KAMA reduces whipsaw in choppy markets (ER drops → KAMA flattens)
- Fisher Transform catches reversals in bear rallies (2022, 2025)
- Choppiness filter switches between trend-follow and mean-revert modes
- 4h TF balances trade frequency vs fee drag

Target: Sharpe>0.45, DD>-35%, trades>=80 train (20/year), trades>=12 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_chop_regime_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market noise via Efficiency Ratio (ER)
    ER=1 → trending (fast SC), ER=0 → choppy (slow SC)
    """
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    # Efficiency Ratio: net change / sum of absolute changes
    er = np.zeros(n)
    er[:] = np.nan
    for i in range(period, n):
        net_change = abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
    
    # Smoothing Constant: SC = [ER * (fast_SC - slow_SC) + slow_SC]^2
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = np.zeros(n)
    sc[:] = np.nan
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[:] = np.nan
    kama[period] = close[period]  # initialize
    for i in range(period + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution, highlights reversals
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Fisher needs high/low midrange
    hl_mid = np.zeros(n)
    hl_mid[:] = np.nan
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        hh = np.max(close[i - period + 1:i + 1])
        ll = np.min(close[i - period + 1:i + 1])
        
        if hh > ll:
            # Normalize to -1 to +1 range
            hl_mid[i] = 0.67 * ((close[i] - ll) / (hh - ll) - 0.5) + 0.67 * hl_mid[i - 1] if i > period else 0.0
            hl_mid[i] = np.clip(hl_mid[i], -0.99, 0.99)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1 + hl_mid[i]) / (1 - hl_mid[i]))
            fisher_prev[i] = fisher[i - 1] if i > period else fisher[i]
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = choppy/range-bound
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        if hh > ll:
            atr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                tr = max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1]))
                atr_sum += tr
            
            chop[i] = 100.0 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    
    return chop

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
    
    # Calculate and align HTF KAMA for trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10)
    kama_4h_fast = calculate_kama(close, period=5)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher(close, period=9)
    sma_100 = calculate_sma(close, 100)
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
        
        if np.isnan(kama_4h[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_100[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = choppy/range (mean revert)
        # CHOP < 38.2 = trending (trend follow)
        # 38.2-61.8 = neutral (use previous regime)
        
        is_trending = chop[i] < 38.2
        is_choppy = chop[i] > 61.8
        
        # === DUAL HTF BIAS (1d + 1w must agree) ===
        htf_1d_bull = close[i] > kama_1d_aligned[i]
        htf_1d_bear = close[i] < kama_1d_aligned[i]
        htf_1w_bull = close[i] > kama_1w_aligned[i]
        htf_1w_bear = close[i] < kama_1w_aligned[i]
        
        htf_both_bull = htf_1d_bull and htf_1w_bull
        htf_both_bear = htf_1d_bear and htf_1w_bear
        
        # === 4h KAMA TREND ===
        kama_bull = close[i] > kama_4h[i]
        kama_bear = close[i] < kama_4h[i]
        
        # === KAMA CROSSOVER ===
        kama_cross_long = False
        kama_cross_short = False
        if i > 0 and not np.isnan(kama_4h_fast[i]) and not np.isnan(kama_4h_fast[i-1]):
            if not np.isnan(kama_4h[i]) and not np.isnan(kama_4h[i-1]):
                if kama_4h_fast[i-1] <= kama_4h[i-1] and kama_4h_fast[i] > kama_4h[i]:
                    kama_cross_long = True
                if kama_4h_fast[i-1] >= kama_4h[i-1] and kama_4h_fast[i] < kama_4h[i]:
                    kama_cross_short = True
        
        # === FISHER TRANSFORM REVERSALS ===
        fisher_cross_long = False
        fisher_cross_short = False
        if not np.isnan(fisher[i]) and not np.isnan(fisher_prev[i]):
            # Long: Fisher crosses above -1.5 (oversold reversal)
            if fisher_prev[i] <= -1.5 and fisher[i] > -1.5:
                fisher_cross_long = True
            # Short: Fisher crosses below +1.5 (overbought reversal)
            if fisher_prev[i] >= 1.5 and fisher[i] < 1.5:
                fisher_cross_short = True
        
        # === SMA FILTER ===
        above_sma100 = close[i] > sma_100[i]
        below_sma100 = close[i] < sma_100[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === ENTRY LOGIC (LOOSE - multiple paths to entry) ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (CHOP < 38.2)
        if is_trending:
            # Long: Dual HTF bull + (KAMA bull OR KAMA cross OR Fisher long)
            if htf_both_bull:
                if kama_bull or kama_cross_long or fisher_cross_long:
                    desired_signal = SIZE_STRONG
            
            # Short: Dual HTF bear + (KAMA bear OR KAMA cross OR Fisher short)
            elif htf_both_bear:
                if kama_bear or kama_cross_short or fisher_cross_short:
                    desired_signal = -SIZE_STRONG
            
            # Single HTF agreement + strong 4h signal
            elif htf_1d_bull and kama_bull and kama_cross_long:
                desired_signal = SIZE_BASE
            elif htf_1d_bear and kama_bear and kama_cross_short:
                desired_signal = -SIZE_BASE
        
        # REGIME 2: CHOPPY (CHOP > 61.8)
        elif is_choppy:
            # Mean reversion: RSI extremes + SMA200 filter
            if rsi_oversold and above_sma200:
                desired_signal = SIZE_BASE
            elif rsi_overbought and below_sma200:
                desired_signal = -SIZE_BASE
            
            # Fisher reversals work well in chop
            elif fisher_cross_long and above_sma100:
                desired_signal = SIZE_BASE
            elif fisher_cross_short and below_sma100:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME (38.2 <= CHOP <= 61.8)
        else:
            # Only take strongest signals in neutral
            if htf_both_bull and kama_bull and rsi[i] < 50:
                desired_signal = SIZE_BASE
            elif htf_both_bear and kama_bear and rsi[i] > 50:
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