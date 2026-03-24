#!/usr/bin/env python3
"""
Experiment #223: 6h Primary + 1d/1w HTF — Fisher Transform + Keltner + KAMA Trend

Hypothesis: 6h timeframe is underexplored and sits between 4h (too noisy) and 12h (too slow).
This strategy combines:

1. KAMA (Kaufman Adaptive Moving Average) - adapts to market noise, reduces whipsaws
2. Ehlers Fisher Transform - catches reversals at extremes, works well in bear/range markets
3. Keltner Channels - volatility-based breakout/reversion levels (different from Donchian/BB)
4. ADX + Choppiness dual regime filter - trend vs mean-reversion mode
5. 1d/1w HTF alignment - only trade when daily AND weekly trend agree (strong confluence)

Entry Logic:
- TREND REGIME (ADX>20, CHOP<55): KAMA direction + HTF alignment + Fisher crossover
- MEAN REVERSION (ADX<20 or CHOP>55): Fade Keltner extremes with Fisher confirmation

Key improvements over failed 6h attempts:
- Less strict than #211 (which had negative Sharpe)
- Fisher Transform proven in bear markets (2022 crash, 2025 bear)
- Keltner channels less prone to false breakouts than Donchian
- Dual HTF (1d+1w) ensures we only trade with major trend

Position sizing: 0.25 base, 0.35 when 1d+1w align (strong confluence)
Stoploss: 2.5x ATR trailing
Target: 30-60 trades/year, Sharpe>0.4 (beat current 6h best of 0.399)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_keltner_kama_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=21, fast_sc=2/11, slow_sc=2/201):
    """Kaufman Adaptive Moving Average - adapts smoothing to market noise"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    for i in range(period, n):
        price_change = abs(close[i] - close[i-period])
        volatility_sum = sum(abs(close[j] - close[j-1]) for j in range(i-period+1, i+1))
        
        if volatility_sum < 1e-10:
            er = 0.0
        else:
            er = price_change / volatility_sum
        
        er = np.clip(er, 0.0, 1.0)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """Ehlers Fisher Transform - normalizes price to identify turning points"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range < 1e-10:
            fisher[i] = 0.0
        else:
            median = (high[i] + low[i]) / 2.0
            normalized = (median - lowest) / price_range
            normalized = np.clip(normalized, 0.001, 0.999)
            fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        if i > period:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

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

def calculate_keltner_channels(high, low, close, ema_period=20, atr_period=14, atr_mult=2.0):
    """Keltner Channels - EMA +/- ATR multiplier"""
    n = len(close)
    if n < max(ema_period, atr_period) + 1:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    
    return upper, lower, ema

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for major trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=34)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for major trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    kama = calculate_kama(close, period=21)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    keltner_upper, keltner_lower, keltner_mid = calculate_keltner_channels(high, low, close, ema_period=20, atr_period=14, atr_mult=2.0)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama[i]) or np.isnan(fisher[i]) or np.isnan(adx[i]) or np.isnan(chop[i]):
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
        
        # === HTF BIAS (1d + 1w) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Strong HTF alignment (both 1d and 1w agree)
        htf_strong_bull = htf_1d_bull and htf_1w_bull
        htf_strong_bear = htf_1d_bear and htf_1w_bear
        
        # === KAMA TREND ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_bull = False
        fisher_bear = False
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_signal[i-1]):
            # Bullish: Fisher crosses above signal line from below
            fisher_bull = fisher[i] > fisher_signal[i] and fisher[i-1] <= fisher_signal[i-1]
            # Bearish: Fisher crosses below signal line from above
            fisher_bear = fisher[i] < fisher_signal[i] and fisher[i-1] >= fisher_signal[i-1]
        
        # === KELTNER BREAKOUT ===
        keltner_breakout_long = close[i] > keltner_upper[i]
        keltner_breakout_short = close[i] < keltner_lower[i]
        
        # === REGIME DETECTION ===
        is_trending = adx[i] > 20.0
        is_weak = adx[i] <= 20.0
        is_choppy = chop[i] > 55.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (ADX > 20, CHOP <= 55)
        if is_trending and not is_choppy:
            # Long: KAMA bull + HTF support + Fisher/Keltner trigger
            if kama_bull and (htf_strong_bull or htf_1d_bull):
                if fisher_bull or keltner_breakout_long:
                    if htf_strong_bull:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            
            # Short: KAMA bear + HTF support + Fisher/Keltner trigger
            elif kama_bear and (htf_strong_bear or htf_1d_bear):
                if fisher_bear or keltner_breakout_short:
                    if htf_strong_bear:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
        # REGIME 2: WEAK/CHOPPY (ADX <= 20 or CHOP > 55) - Mean Reversion
        elif is_weak or is_choppy:
            # Fade Keltner extremes with Fisher confirmation + HTF alignment
            if keltner_breakout_long and fisher_bear:
                if htf_1d_bear:
                    desired_signal = -SIZE_BASE * 0.6
                else:
                    desired_signal = -SIZE_BASE * 0.4
            elif keltner_breakout_short and fisher_bull:
                if htf_1d_bull:
                    desired_signal = SIZE_BASE * 0.6
                else:
                    desired_signal = SIZE_BASE * 0.4
        
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
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
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
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals