#!/usr/bin/env python3
"""
Experiment #1422: 12h Primary + 1d HTF — KAMA Adaptive Trend + RSI Pullback + Choppiness Filter

Hypothesis: 12h timeframe with 1d HTF trend filter will outperform 4h and 1d in bear/range markets.
Based on research patterns that worked:
1. KAMA (Kaufman Adaptive Moving Average) adapts to volatility — slows in chop, speeds in trend
2. RSI(14) pullback entries within trend direction catch better risk/reward than breakouts
3. Choppiness Index filters out range markets where trend strategies fail
4. 1d HMA(21) provides cleaner trend signal than 1w (less lag, more responsive)

Why 12h not 4h or 1d:
- 4h strategies consistently fail (#1414, #1419, #1421 all negative Sharpe)
- 1d has too few signals in some periods (#1417 Sharpe=-0.416)
- 12h is the "goldilocks" zone — fewer false signals than 4h, more trades than 1d
- Target: 25-40 trades/year (fee drag ~1.5-2%)

Design:
1. 1d HMA(21) = trend direction (call ONCE before loop via mtf_data)
2. KAMA(10,2,30) on 12h = adaptive trend following (ER-based smoothing)
3. RSI(14) = pullback entry trigger (long: 35-45, short: 55-65)
4. Choppiness(14) > 55 = skip trend entries (range market)
5. ATR(14) trailing stop 2.5x = risk management
6. Position size 0.30 = conservative for 12h volatility

Key improvements over #1417:
- KAMA instead of HMA on primary TF (adapts to regime automatically)
- RSI pullback instead of CRSI (simpler, more reliable in trends)
- 1d HTF instead of 1w (more responsive to trend changes)
- Tighter Choppiness threshold (55 vs 61.8) for more trades

Target: Sharpe > 0.618 (beat current best), trades >= 30 train, >= 5 test, DD > -40%
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_pullback_1d_hma_chop_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_kama(close, eff_ratio_period=10, fast_sc=2, slow_sc=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    ER = |net_change| / sum(abs(individual_changes))
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    if n < eff_ratio_period + 1:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(eff_ratio_period, n):
        net_change = abs(close[i] - close[i - eff_ratio_period])
        sum_changes = np.sum(np.abs(np.diff(close[i - eff_ratio_period:i + 1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant
    sc = np.full(n, np.nan)
    fast_const = 2.0 / (fast_sc + 1)
    slow_const = 2.0 / (slow_sc + 1)
    
    for i in range(eff_ratio_period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_const - slow_const) + slow_const) ** 2
    
    # Calculate KAMA
    kama[eff_ratio_period] = close[eff_ratio_period]
    for i in range(eff_ratio_period + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            tr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            
            chop[i] = 100.0 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[j] - close[j-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    kama = calculate_kama(close, eff_ratio_period=10, fast_sc=2, slow_sc=30)
    rsi = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1d HMA) - direction filter ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND (12h) - adaptive trend confirmation ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === CHOPPINESS REGIME FILTER ===
        is_choppy = chop[i] > 55.0  # Range market - avoid trend entries
        is_trending = chop[i] < 55.0  # Trending market - allow entries
        
        # === RSI PULLBACK ZONES (within trend) ===
        # Long: RSI pulled back to 35-45 zone in uptrend
        rsi_pullback_long = 35.0 <= rsi[i] <= 48.0
        # Short: RSI pulled back to 52-65 zone in downtrend
        rsi_pullback_short = 52.0 <= rsi[i] <= 65.0
        
        # RSI extreme oversold/overbought (stronger signal)
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === DESIRED SIGNAL - PULLBACK WITHIN TREND ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        # Path 1: Trending regime + RSI pullback + macro bull + KAMA bull
        if is_trending and rsi_pullback_long and macro_bull and kama_bull:
            desired_signal = BASE_SIZE
        # Path 2: Trending regime + RSI oversold + macro bull + KAMA bull (stronger)
        elif is_trending and rsi_oversold and macro_bull and kama_bull:
            desired_signal = BASE_SIZE
        # Path 3: Choppy regime + RSI very oversold + macro bull (mean reversion)
        elif is_choppy and rsi[i] < 30.0 and macro_bull:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRIES
        # Path 1: Trending regime + RSI pullback + macro bear + KAMA bear
        if is_trending and rsi_pullback_short and macro_bear and kama_bear:
            desired_signal = -BASE_SIZE
        # Path 2: Trending regime + RSI overbought + macro bear + KAMA bear (stronger)
        elif is_trending and rsi_overbought and macro_bear and kama_bear:
            desired_signal = -BASE_SIZE
        # Path 3: Choppy regime + RSI very overbought + macro bear (mean reversion)
        elif is_choppy and rsi[i] > 70.0 and macro_bear:
            desired_signal = -BASE_SIZE * 0.5
        
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
        if abs(desired_signal) >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals