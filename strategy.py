#!/usr/bin/env python3
"""
EXPERIMENT #009 - DEMA Crossover + 4h HMA Trend + BB Regime Filter (1h primary)
================================================================================
Hypothesis: DEMA (Double EMA) provides faster trend detection than HMA while 
reducing lag. Combined with 4h HMA(50) for major trend alignment and Bollinger 
Band Width regime filter to avoid choppy markets. This differs from failed 
strategies by using DEMA instead of HMA/KAMA for primary signals, with stricter 
regime filtering (BB Width > 60th percentile).

Key features:
- Primary TF: 1h (as required for Experiment #009)
- HTF filter: 4h HMA(50) for major trend direction
- Trend: DEMA(8/21) crossover on 1h
- Regime: Bollinger Band Width > 60th percentile (trending market only)
- Volume filter: Volume > 1.5x 20-period average (confirms breakout)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit, trail stop at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "dema_crossover_regime_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_dema(close, period):
    """Calculate Double Exponential Moving Average"""
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    dema = 2 * ema1 - ema2
    return dema.values


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
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


def calculate_bollinger_bands(close, period=20, std_dev=2):
    """Calculate Bollinger Bands and Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    band_width = (upper - lower) / sma
    return upper.values, lower.values, band_width.values


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
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 50)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    dema_fast = calculate_dema(close, 8)
    dema_slow = calculate_dema(close, 21)
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2)
    
    # Calculate Bollinger Band Width percentile rank (regime filter)
    bb_width_pr = calculate_percentile_rank(bb_width, 100)
    
    # Calculate volume average for volume filter
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(dema_fast[i]) or 
            np.isnan(dema_slow[i]) or np.isnan(atr[i]) or 
            np.isnan(bb_width_pr[i]) or np.isnan(volume_avg[i]) or 
            atr[i] == 0 or volume_avg[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter (HTF)
        htf_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # 1h DEMA crossover signal
        dema_signal = 0
        if dema_fast[i] > dema_slow[i]:
            dema_signal = 1
        elif dema_fast[i] < dema_slow[i]:
            dema_signal = -1
        
        # Regime filter: only trade when BB Width is in top 60% (trending market)
        regime_valid = bb_width_pr[i] > 0.60
        
        # Volume filter: volume > 1.5x average
        volume_valid = volume[i] > 1.5 * volume_avg[i]
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: DEMA bullish crossover + 4h trend bullish + Regime valid + Volume confirmation
        if dema_signal == 1 and htf_trend == 1 and regime_valid and volume_valid:
            target_signal = SIZE
        
        # Short entry: DEMA bearish crossover + 4h trend bearish + Regime valid + Volume confirmation
        elif dema_signal == -1 and htf_trend == -1 and regime_valid and volume_valid:
            target_signal = -SIZE
        
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
                
                # Check take profit (2R from entry, where R = 2.5*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * atr[i]:  # 2R = 5*ATR
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
                    if close[i] <= entry_price - 5.0 * atr[i]:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
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
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                if position_side == 1 and dema_signal == -1:
                    # DEMA crossed bearish, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and dema_signal == 1:
                    # DEMA crossed bullish, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals