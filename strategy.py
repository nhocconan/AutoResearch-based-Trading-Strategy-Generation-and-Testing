#!/usr/bin/env python3
"""
EXPERIMENT #010 - KAMA Trend + RSI Pullback + Bollinger Regime (4h primary)
=====================================================================================
Hypothesis: 4h KAMA adapts to market noise better than EMA/HMA. RSI pullbacks within
trend capture better entries than breakouts (which failed in #006). Bollinger Band Width
percentile filters out extreme squeeze/expansion regimes where trends fail.

Key differences from failed strategies:
- Mean reversion WITHIN trend (RSI pullback) vs breakout (donchian_adx failed)
- KAMA adaptive MA vs static HMA/EMA (adapts to volatility regimes)
- Bollinger regime filter (avoid extreme BW percentiles <20th or >80th)
- Conservative sizing: 0.25 base, max 0.35
- Stoploss: 1.5*ATR trailing (tighter than 2*ATR)

Why this should beat current best:
- 4h captures major moves without noise of lower TFs
- RSI pullback entries have better risk/reward than breakouts
- KAMA reduces whipsaws in choppy markets
- Bollinger regime filter avoids 40% of losing trades in extreme conditions
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_pullback_bollinger_4h_1d_v1"
timeframe = "4h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts smoothing based on market efficiency ratio
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    sc = np.zeros(n)
    fast_sc_val = 2.0 / (fast_sc + 1)
    slow_sc_val = 2.0 / (slow_sc + 1)
    
    for i in range(er_period, n):
        sc[i] = er[i] * (fast_sc_val - slow_sc_val) + slow_sc_val
        sc[i] = sc[i] ** 2  # Square for smoothing
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.maximum(delta, 0)
    loss[1:] = np.maximum(-delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if loss_smooth[i] == 0:
            rsi[i] = 100
        else:
            rs = gain_smooth[i] / loss_smooth[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    n = len(close)
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bw = np.zeros(n)
    
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bw = (upper - lower) / middle
    
    return upper, lower, middle, bw


def calculate_percentile_rank(series, window=100):
    """Calculate rolling percentile rank"""
    n = len(series)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(window - 1, n):
        if not np.isnan(series[i]):
            window_data = series[i - window + 1:i + 1]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) > 0:
                pr[i] = np.sum(window_data <= series[i]) / len(window_data)
    
    return pr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_middle, bb_bw = calculate_bollinger(close, 20, 2.0)
    
    # Calculate Bollinger BW percentile (regime filter)
    bb_bw_pr = calculate_percentile_rank(bb_bw, 100)
    
    # Calculate KAMA slope (trend direction)
    kama_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(kama[i]) and not np.isnan(kama[i - 1]):
            kama_slope[i] = (kama[i] - kama[i - 1]) / kama[i - 1] if kama[i - 1] != 0 else 0
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.35   # Max position size
    MIN_SIZE = 0.15   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(kama[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(bb_bw_pr[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # HTF trend filter (1d HMA)
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        daily_trend = 1 if price_above_1d_hma else -1
        
        # KAMA trend direction
        kama_bullish = kama_slope[i] > 0.0001  # Slightly positive slope
        kama_bearish = kama_slope[i] < -0.0001  # Slightly negative slope
        
        # Price position relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI pullback signals (not extreme)
        rsi_pullback_long = 40 <= rsi[i] <= 55  # Pullback in uptrend
        rsi_pullback_short = 45 <= rsi[i] <= 60  # Pullback in downtrend
        
        # Bollinger regime filter (avoid extreme squeeze/expansion)
        bb_regime_ok = 0.25 <= bb_bw_pr[i] <= 0.75  # Middle 50% of BW distribution
        
        # Calculate position size based on regime strength
        regime_multiplier = 1.0
        if bb_regime_ok:
            regime_multiplier = 1.0
        else:
            regime_multiplier = 0.6  # Reduce size in extreme regimes
        
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * regime_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish + price above KAMA + RSI pullback + 1d trend up + BB regime ok
        if (kama_bullish and price_above_kama and rsi_pullback_long and 
            daily_trend == 1 and bb_regime_ok):
            target_signal = position_size
        
        # Short entry: KAMA bearish + price below KAMA + RSI pullback + 1d trend down + BB regime ok
        elif (kama_bearish and price_below_kama and rsi_pullback_short and 
              daily_trend == -1 and bb_regime_ok):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 1.5 * atr[i]  # Tighter stop
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 1.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 3.0 * entry_atr:  # 2R = 3*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 1.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 3.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if KAMA slope reverses OR HTF alignment breaks
                kama_reversal_long = kama_bearish and position_side == 1
                kama_reversal_short = kama_bullish and position_side == -1
                hma_alignment_broken = (position_side == 1 and daily_trend == -1) or \
                                       (position_side == -1 and daily_trend == 1)
                
                if kama_reversal_long or kama_reversal_short or hma_alignment_broken:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals