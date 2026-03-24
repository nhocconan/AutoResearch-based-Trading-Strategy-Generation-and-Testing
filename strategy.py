#!/usr/bin/env python3
"""
Experiment #668: 4h Primary + 12h/1d HTF — Dual Regime KAMA + RSI Pullback

Hypothesis: Previous KAMA+Donchian strategies failed due to overly strict entry conditions
(0 trades). This version SIMPLIFIES entries while maintaining edge:
1. 12h HMA for primary trend bias (more responsive than 1d)
2. KAMA(14) slope for 4h momentum direction
3. RSI(7) pullback entries (not breakouts) - enters on dips in uptrend, rallies in downtrend
4. ADX(14) > 18 filter (lowered from 20 to allow more trades)
5. Choppiness Index for regime detection - reduces size in chop, full size in trend
6. ATR(14) 2.5x trailing stop for risk management

Key changes from #638:
- RSI pullback instead of Donchian breakout (more frequent entries)
- 12h HMA instead of 1d (more responsive to trend changes)
- Lower ADX threshold (18 vs 20) for more trades
- Simpler entry logic - fewer confluence requirements
- Size modulation by chop vs trend regime

Target: Sharpe>0.40, trades>=30 train, trades>=3 test
Timeframe: 4h
Size: 0.20-0.35 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_pullback_chop_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=14, fast=2, slow=30):
    """Kaufman Adaptive Moving Average - adjusts smoothing based on market efficiency"""
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        
        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            volatility += abs(close[j] - close[j - 1])
        
        if volatility > 1e-10:
            er = price_change / volatility
        else:
            er = 0.0
        
        sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
        
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0.0)
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    rs[:] = np.nan
    mask = avg_loss > 1e-10
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0.0)
        else:
            plus_dm[i] = 0.0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0.0)
        else:
            minus_dm[i] = 0.0
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

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
    """Hull Moving Average for HTF"""
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market chop vs trend"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high - lowest_low > 1e-10:
            tr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            
            chop[i] = 100.0 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 4h indicators
    kama = calculate_kama(close, period=14, fast=2, slow=30)
    rsi = calculate_rsi(close, period=7)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30
    SIZE_CHOP = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h HMA) ===
        htf_bull = close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_12h_aligned[i]
        
        # === KAMA TREND DIRECTION (simplified - just current vs 2 bars ago) ===
        kama_bull = kama[i] > kama[i-2] if i >= 2 and not np.isnan(kama[i-2]) else False
        kama_bear = kama[i] < kama[i-2] if i >= 2 and not np.isnan(kama[i-2]) else False
        
        # === ADX TREND STRENGTH ===
        trend_strong = adx[i] > 18.0
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 50.0
        is_trending = chop[i] < 45.0
        
        # === RSI PULLBACK LEVELS ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        rsi_extreme_oversold = rsi[i] < 30.0
        rsi_extreme_overbought = rsi[i] > 70.0
        
        # === ENTRY LOGIC (SIMPLIFIED - LOOSE CONDITIONS) ===
        desired_signal = 0.0
        size_multiplier = 1.0
        
        if is_trending:
            size_multiplier = 1.0
        elif is_choppy:
            size_multiplier = 0.67  # Reduce size in chop
        
        # LONG entries (HTF bull + KAMA up + RSI pullback)
        if htf_bull and kama_bull:
            if trend_strong and rsi_oversold:
                desired_signal = SIZE_TREND * size_multiplier
            elif rsi_extreme_oversold:
                # Extreme RSI override - enter even without ADX confirmation
                desired_signal = SIZE_TREND * size_multiplier
            elif close[i] > kama[i] and rsi[i] < 50:
                # Weaker signal: price above KAMA + RSI below midpoint
                desired_signal = SIZE_CHOP * size_multiplier
        
        # SHORT entries (HTF bear + KAMA down + RSI rally)
        elif htf_bear and kama_bear:
            if trend_strong and rsi_overbought:
                desired_signal = -SIZE_TREND * size_multiplier
            elif rsi_extreme_overbought:
                # Extreme RSI override
                desired_signal = -SIZE_TREND * size_multiplier
            elif close[i] < kama[i] and rsi[i] > 50:
                # Weaker signal: price below KAMA + RSI above midpoint
                desired_signal = -SIZE_CHOP * size_multiplier
        
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
        if desired_signal >= SIZE_TREND * 0.9:
            final_signal = SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.9:
            final_signal = -SIZE_TREND
        elif desired_signal >= SIZE_CHOP * 0.9:
            final_signal = SIZE_CHOP
        elif desired_signal <= -SIZE_CHOP * 0.9:
            final_signal = -SIZE_CHOP
        elif abs(desired_signal) > 0.05:
            final_signal = np.sign(desired_signal) * SIZE_CHOP
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