#!/usr/bin/env python3
"""
EXPERIMENT #088 - HMA/KAMA Adaptive Trend + Bollinger Regime + Dual HTF Filter (4h primary)
============================================================================================
Hypothesis: 4h timeframe captures medium-term trends well. HMA provides fast trend detection,
KAMA adapts to volatility (flattens in chop, accelerates in trends). Bollinger Band Width
percentile detects regime (squeeze=chop, expansion=trend). Dual HTF (1d/1w HMA) ensures
we trade with major trend. Pullback entries to KAMA reduce false breakouts.

Key features:
- Primary TF: 4h
- HTF filters: 1d HMA(50) + 1w HMA(50) for trend alignment
- Trend: HMA(21) vs HMA(50) crossover + KAMA(14) confirmation
- Regime: Bollinger Band Width percentile > 40th (avoid extreme squeeze)
- Entry: Pullback to KAMA in direction of HMA trend
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete, scaled by BBW regime strength

Why this should beat current best (Sharpe=0.490):
- KAMA adapts to volatility better than fixed MAs
- BBW regime filter avoids trading in extreme compression
- 4h captures more signals than 12h while avoiding 15m/30m noise
- Pullback entries have better risk/reward than breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_kama_bbwregime_dualhtf_4h_1d_1w_v1"
timeframe = "4h"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_kama(close, high, low, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)"""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period - 1, n):
        price_change = abs(close[i] - close[i - er_period + 1])
        volatility = np.sum(np.abs(np.diff(close[i - er_period + 1:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # Initialize KAMA
    kama[er_period - 1] = close[er_period - 1]
    
    # Calculate KAMA
    for i in range(er_period, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma, std


def calculate_bbw_percentile(bbw, window=100):
    """Calculate rolling percentile rank of Bollinger Band Width"""
    n = len(bbw)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(window - 1, n):
        if not np.isnan(bbw[i]):
            window_data = bbw[i - window + 1:i + 1]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) > 0:
                pr[i] = np.sum(window_data <= bbw[i]) / len(window_data)
    
    return pr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1w = calculate_hma(df_1w['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    kama = calculate_kama(close, high, low, er_period=10)
    atr = calculate_atr(high, low, close, 14)
    
    # Bollinger Bands for regime detection
    bb_upper, bb_lower, bb_sma, bb_std = calculate_bollinger_bands(close, 20, 2.0)
    bbw = (bb_upper - bb_lower) / bb_sma  # Band Width
    bbw_pr = calculate_bbw_percentile(bbw, 100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size
    MIN_SIZE = 0.20   # Min position size
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
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or
            np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or np.isnan(kama[i]) or
            np.isnan(atr[i]) or np.isnan(bbw_pr[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Dual HTF trend alignment
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        
        # 1d and 1w trend direction
        daily_trend = 1 if price_above_1d_hma else -1
        weekly_trend = 1 if price_above_1w_hma else -1
        
        # 4h HMA trend
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # KAMA position relative to price (pullback detection)
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Bollinger Band Width regime (avoid extreme squeeze)
        bbw_regime_ok = bbw_pr[i] > 0.30  # Not in extreme compression
        
        # Calculate position size based on BBW regime strength
        bbw_multiplier = min(1.0 + (bbw_pr[i] - 0.30) * 0.5, 1.25)  # Max 1.25x
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * bbw_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: HMA bullish + KAMA pullback + HTF alignment + BBW regime OK
        # Entry when price pulls back to KAMA but HMA trend is still bullish
        if (hma_bullish and price_above_kama and daily_trend == 1 and 
            weekly_trend == 1 and bbw_regime_ok):
            # Additional confirmation: price not too extended above KAMA
            kama_distance = (close[i] - kama[i]) / kama[i] * 100
            if kama_distance < 3.0:  # Within 3% of KAMA
                target_signal = position_size
        
        # Short entry: HMA bearish + KAMA pullback + HTF alignment + BBW regime OK
        elif (hma_bearish and price_below_kama and daily_trend == -1 and 
              weekly_trend == -1 and bbw_regime_ok):
            # Additional confirmation: price not too extended below KAMA
            kama_distance = (kama[i] - close[i]) / kama[i] * 100
            if kama_distance < 3.0:  # Within 3% of KAMA
                target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2.5R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 6.25 * entry_atr:  # 2.5R = 6.25*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 6.25 * entry_atr:  # 2.5R profit
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
            # Reduce position to half at 2.5R profit
            signals[i] = HALF_SIZE * np.sign(position_side)
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
                # Exit if HMA crossover reverses OR HTF alignment breaks
                hma_reversal_long = hma_bearish  # Was long, now HMA bearish
                hma_reversal_short = hma_bullish  # Was short, now HMA bullish
                hma_alignment_broken = (position_side == 1 and daily_trend == -1) or \
                                       (position_side == -1 and daily_trend == 1)
                
                if hma_reversal_long or hma_reversal_short or hma_alignment_broken:
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