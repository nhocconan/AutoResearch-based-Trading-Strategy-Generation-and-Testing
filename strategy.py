#!/usr/bin/env python3
"""
EXPERIMENT #068 - KAMA Adaptive Trend + Bollinger Regime + Dual HTF Filter (30m primary)
=========================================================================================
Hypothesis: 30m KAMA crossovers capture medium-term trends with adaptive volatility response.
Most failures occur in choppy regimes - Bollinger Band Width percentile detects expansion
(trend) vs contraction (chop). Dual HTF filter (4h HMA + 1d HMA) ensures alignment with
major trend direction. This differs from failed strategies by using KAMA (adaptive) instead
of static EMA/HMA, adding BB regime filter to avoid squeeze periods, and conservative sizing.

Key features:
- Primary TF: 30m
- HTF filters: 4h HMA(21) + 1d HMA(50) for dual alignment
- Trend: KAMA(10) vs KAMA(30) crossover (adaptive to volatility)
- Regime: Bollinger Band Width percentile > 50th (avoid squeeze/chop)
- Entry: KAMA cross + HTF alignment + BB regime expansion + volume confirmation
- Stoploss: 2.5*ATR(14) trailing (wider for 30m noise)
- Position sizing: 0.25 base, 0.30 max with strong signals, discrete levels
- Take profit: Reduce to half at 2.5R profit

Why this should beat current best (Sharpe=0.490):
- KAMA adapts to volatility - faster in trends, slower in chop
- BB regime filter removes 40%+ of trades during squeeze periods
- Dual HTF (4h+1d) simpler than triple but still effective
- Conservative sizing (0.25-0.30) controls drawdown better than 0.35
- Wider stops (2.5*ATR vs 2.0*ATR) reduce whipsaw exits on 30m
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_bbregime_dualhtf_30m_4h_1d_v1"
timeframe = "30m"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period - 1, n):
        change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma


def calculate_bb_width(upper, lower, sma):
    """Calculate Bollinger Band Width"""
    width = np.zeros(len(upper))
    for i in range(len(upper)):
        if sma[i] > 0:
            width[i] = (upper[i] - lower[i]) / sma[i]
        else:
            width[i] = 0
    return width


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
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    kama_fast = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, period=30, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, 14)
    
    # Bollinger Bands for regime detection
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_sma)
    bb_width_pr = calculate_percentile_rank(bb_width, 100)
    
    # Volume SMA for confirmation
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size with strong signals
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
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]) or
            np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]) or
            np.isnan(atr[i]) or np.isnan(bb_width_pr[i]) or
            np.isnan(volume_sma[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Dual HTF trend alignment
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # 4h and 1d trend direction
        htf_trend = 0
        if price_above_4h_hma and price_above_1d_hma:
            htf_trend = 1  # Bullish alignment
        elif not price_above_4h_hma and not price_above_1d_hma:
            htf_trend = -1  # Bearish alignment
        else:
            htf_trend = 0  # Mixed/neutral
        
        # Bollinger Band regime (avoid squeeze)
        bb_expanding = bb_width_pr[i] > 0.50  # In top 50% of bandwidth
        
        # Volume confirmation
        volume_above_avg = volume[i] > volume_sma[i]
        
        # KAMA crossover signals
        kama_bullish_cross = (kama_fast[i] > kama_slow[i] and 
                              kama_fast[i - 1] <= kama_slow[i - 1])
        kama_bearish_cross = (kama_fast[i] < kama_slow[i] and 
                              kama_fast[i - 1] >= kama_slow[i - 1])
        
        # KAMA trend confirmation (already in position)
        kama_trend_long = kama_fast[i] > kama_slow[i]
        kama_trend_short = kama_fast[i] < kama_slow[i]
        
        # Calculate position size based on signal strength
        position_size = BASE_SIZE
        if htf_trend != 0 and bb_expanding and volume_above_avg:
            position_size = MAX_SIZE
        elif htf_trend != 0 and bb_expanding:
            position_size = BASE_SIZE + 0.025
        else:
            position_size = MIN_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish cross + HTF bullish + BB expanding + volume
        if (kama_bullish_cross and htf_trend == 1 and bb_expanding and volume_above_avg):
            target_signal = position_size
        
        # Short entry: KAMA bearish cross + HTF bearish + BB expanding + volume
        elif (kama_bearish_cross and htf_trend == -1 and bb_expanding and volume_above_avg):
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
                # Exit if KAMA reverses OR HTF alignment breaks OR BB contracts severely
                kama_reversal_long = kama_fast[i] < kama_slow[i]
                kama_reversal_short = kama_fast[i] > kama_slow[i]
                hma_alignment_broken = (position_side == 1 and htf_trend == -1) or \
                                       (position_side == -1 and htf_trend == 1)
                bb_squeeze = bb_width_pr[i] < 0.20  # Severe contraction
                
                if kama_reversal_long or kama_reversal_short or hma_alignment_broken or bb_squeeze:
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