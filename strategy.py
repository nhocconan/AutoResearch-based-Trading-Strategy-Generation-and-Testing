#!/usr/bin/env python3
"""
Experiment #407: 6h Primary + 1d/1w HTF — KAMA/Fisher Dual Regime v1

Hypothesis: Previous 6h strategies failed due to (1) overly complex entry conditions
resulting in 0 trades, or (2) using HMA/RSI which whipsaw in 6h regime changes.
This version uses KAMA (Kaufman Adaptive Moving Average) which adjusts to volatility,
plus Ehlers Fisher Transform for cleaner reversal signals than RSI.

Key innovations:
1. KAMA adapts smoothing based on market noise ratio - fewer false signals in chop
2. Fisher Transform normalizes price to Gaussian distribution - cleaner extremes than RSI
3. Dual HTF: 1d for trend direction, 1w for major bias (only trade with weekly trend)
4. Choppiness Index regime filter - trending vs mean-reverting logic
5. SIMPLIFIED entries: max 3 confluence conditions to ensure trade generation

Regime Detection:
- CHOP < 38.2 + ADX > 25 = trending → KAMA breakout entries
- CHOP > 61.8 + ADX < 20 = choppy → Fisher mean reversion
- Otherwise = neutral → flat or reduced size

Entry Logic:
- Trending Long: KAMA bull + 1d KAMA bull + 1w KAMA bull + price > KAMA
- Trending Short: KAMA bear + 1d KAMA bear + 1w KAMA bear + price < KAMA
- Choppy Long: Fisher < -1.5 + price > SMA200 (just 2 conditions!)
- Choppy Short: Fisher > +1.5 + price < SMA200 (just 2 conditions!)

Position sizing: 0.25 base, 0.30 when all HTF aligned
Stoploss: 2.5x ATR(14) from entry price

Target: Sharpe>0.45, DD>-35%, trades>=30 train, trades>=5 test, ALL symbols positive
Timeframe: 6h (30-60 trades/year target)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_fisher_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio (ER)
    ER = |price change| / sum of individual price changes
    High ER = trending (fast smoothing), Low ER = choppy (slow smoothing)
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        
        if noise > 1e-10:
            er[i] = price_change / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant (SC)
    # SC = [ER * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)]^2
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Initialize KAMA with SMA
    kama[period] = np.mean(close[:period + 1])
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for cleaner reversal signals
    Based on (2 * ((price - min) / (max - min)) - 1) then Fisher transform
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            fisher[i] = 0.0
            continue
        
        # Normalize price to -1 to +1 range
        normalized = 2.0 * ((high[i] + low[i]) / 2.0 - lowest) / price_range - 1.0
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform: 0.5 * ln((1 + x) / (1 - x))
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
    
    return fisher

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

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
    
    # Calculate primary (6h) indicators
    kama_6h = calculate_kama(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    fisher = calculate_fisher_transform(high, low, period=9)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Regime memory for hysteresis
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
        
        if np.isnan(kama_6h[i]) or np.isnan(fisher[i]) or np.isnan(adx[i]):
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
        
        if np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION with ADX + CHOP ===
        # Trending: ADX > 25 AND CHOP < 38.2
        # Choppy: ADX < 20 AND CHOP > 61.8
        # Otherwise: use previous regime (hysteresis)
        
        is_trending = adx[i] > 25.0 and (np.isnan(chop[i]) or chop[i] < 38.2)
        is_choppy = adx[i] < 20.0 and (np.isnan(chop[i]) or chop[i] > 61.8)
        
        if is_trending:
            current_regime = 1
        elif is_choppy:
            current_regime = 2
        else:
            current_regime = prev_regime
        
        prev_regime = current_regime
        
        # === HTF BIAS (1d and 1w) ===
        htf_1d_bull = close[i] > kama_1d_aligned[i]
        htf_1d_bear = close[i] < kama_1d_aligned[i]
        
        htf_1w_bull = close[i] > kama_1w_aligned[i]
        htf_1w_bear = close[i] < kama_1w_aligned[i]
        
        # === 6h KAMA TREND ===
        kama_bull = close[i] > kama_6h[i]
        kama_bear = close[i] < kama_6h[i]
        
        # === KAMA SLOPE (trend momentum) ===
        kama_slope_bull = False
        kama_slope_bear = False
        if i > 1 and not np.isnan(kama_6h[i-1]):
            if kama_6h[i] > kama_6h[i-1]:
                kama_slope_bull = True
            if kama_6h[i] < kama_6h[i-1]:
                kama_slope_bear = True
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === FISHER EXTREMES (mean reversion signals) ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # === ENTRY LOGIC (SIMPLIFIED - ensure trade generation) ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (KAMA breakout with HTF alignment)
        if current_regime == 1:
            # Long: 6h KAMA bull + 1d KAMA bull + 1w aligned OR neutral
            if kama_bull and htf_1d_bull:
                if htf_1w_bull or not htf_1w_bear:  # 1w not bearish
                    if kama_slope_bull:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            
            # Short: 6h KAMA bear + 1d KAMA bear + 1w aligned OR neutral
            elif kama_bear and htf_1d_bear:
                if htf_1w_bear or not htf_1w_bull:  # 1w not bullish
                    if kama_slope_bear:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
        # REGIME 2: CHOPPY (Fisher mean reversion - VERY SIMPLE)
        elif current_regime == 2:
            # Long: Fisher oversold + above SMA200 (just 2 conditions!)
            if fisher_oversold and above_sma200:
                desired_signal = SIZE_BASE
            
            # Short: Fisher overbought + below SMA200 (just 2 conditions!)
            elif fisher_overbought and below_sma200:
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