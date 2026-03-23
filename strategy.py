#!/usr/bin/env python3
"""
Experiment #1204: 4h Primary + 12h/1d HTF — Fisher Transform + KAMA Adaptive Trend

Hypothesis: Current best (#1179 Sharpe=0.612) uses complex CRSI+Choppiness+Donchian which may be
too restrictive in certain regimes. This version simplifies using:
- Ehlers Fisher Transform (period=9) for reversal detection — proven in bear/range markets
- KAMA (Kaufman Adaptive MA) for trend — adapts to volatility automatically
- ADX (14) for trend strength filter — clearer than Choppiness
- Simpler entry: Fisher extreme + KAMA alignment + ADX confirmation

Why this should work better:
1. Fisher Transform catches reversals at extremes (better than CRSI in 2022 crash)
2. KAMA adapts to volatility — faster in trends, slower in chop
3. ADX > 25 = trend, ADX < 20 = range — cleaner regime separation
4. Fewer entry conditions = more trades (avoid 0-trade failure)
5. 4h timeframe proven to work (20-50 trades/year target)

Position Size: 0.30 discrete
Stoploss: 2.5x ATR trailing
Target: Sharpe > 0.612, trades >= 30 train, >= 3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_adx_12h1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average — adapts smoothing based on market efficiency.
    More responsive in trends, smoother in chop.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        net_change = abs(close[i] - close[i - er_period])
        sum_changes = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform — normalizes price to Gaussian distribution.
    Catches reversals at extreme values (+/- 1.5 to +/- 2.0).
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period:
        return fisher, fisher_signal
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        highest = np.max(window)
        lowest = np.min(window)
        
        if highest - lowest > 1e-10:
            # Normalize price to -1 to +1 range
            normalized = 2.0 * ((close[i] - lowest) / (highest - lowest)) - 1.0
            # Clamp to avoid division issues
            normalized = np.clip(normalized, -0.999, 0.999)
            # Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
            
            if i >= period:
                fisher_signal[i] = fisher[i - 1]
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — measures trend strength.
    ADX > 25 = trending, ADX < 20 = ranging.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    # Calculate True Range and DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth TR and DM
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI and DX
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    mask = tr_smooth > 1e-10
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    for i in range(period * 2 - 1, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX is smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h KAMA for intermediate trend
    kama_12h_raw = calculate_kama(df_12h['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    atr = calculate_atr(high, low, close, period=14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher(close, period=9)
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(adx[i]) or np.isnan(fisher[i]):
            continue
        if np.isnan(kama_4h[i]) or np.isnan(kama_12h_aligned[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA50) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (12h KAMA) ===
        inter_bull = close[i] > kama_12h_aligned[i]
        inter_bear = close[i] < kama_12h_aligned[i]
        
        # === PRIMARY TREND (4h KAMA) ===
        primary_bull = close[i] > kama_4h[i]
        primary_bear = close[i] < kama_4h[i]
        
        # === TREND STRENGTH (ADX) ===
        is_trending = adx[i] > 25.0
        is_ranging = adx[i] < 20.0
        
        # === FISHER TRANSFORM EXTREMES ===
        fisher_oversold = fisher[i] < -1.5 and fisher_signal[i] < fisher[i]
        fisher_overbought = fisher[i] > 1.5 and fisher_signal[i] > fisher[i]
        
        # === DI CROSSOVER ===
        di_bull = plus_di[i] > minus_di[i]
        di_bear = plus_di[i] < minus_di[i]
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === TRENDING REGIME: Follow trend with Fisher pullback entry ===
        if is_trending:
            # Long: bullish alignment + Fisher pullback from oversold
            if macro_bull and inter_bull and primary_bull and fisher_oversold:
                desired_signal = BASE_SIZE
            # Short: bearish alignment + Fisher pullback from overbought
            elif macro_bear and inter_bear and primary_bear and fisher_overbought:
                desired_signal = -BASE_SIZE
        
        # === RANGING REGIME: Mean reversion with Fisher extremes ===
        elif is_ranging:
            # Long: oversold Fisher + not strongly bearish
            if fisher_oversold and not macro_bear:
                desired_signal = BASE_SIZE
            # Short: overbought Fisher + not strongly bullish
            elif fisher_overbought and not macro_bull:
                desired_signal = -BASE_SIZE
        
        # === TRANSITION ZONE: Use DI + Fisher ===
        else:
            if di_bull and fisher_oversold:
                desired_signal = BASE_SIZE
            elif di_bear and fisher_overbought:
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals