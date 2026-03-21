#!/usr/bin/env python3
"""
EXPERIMENT #046 - KAMA Adaptive Trend + 1D Filter (4h primary, 1d HTF)
========================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility
better than fixed MAs. On 4h timeframe, KAMA slope identifies trend direction
while automatically reducing sensitivity during chop. 1d HMA(50) provides major
trend alignment filter. Simple ATR trailing stop with 2R take profit.

Key differences from failed strategies:
- Simpler filter chain (KAMA slope + 1d HMA only, no RSI overfiltering)
- Discrete signal levels (0.0, ±0.25) to minimize fee churn
- Proper MTF loading via mtf_data helper (get_htf_data ONCE before loop)
- Conservative position sizing (25% max) with 2*ATR stoploss

Performance targets:
- Sharpe > 0.315 (beat current best)
- Drawdown < -30%
- Trades > 50
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_trend_4h_1d_v1"
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


def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
    
    # Initialize first KAMA value
    valid_start = period
    while valid_start < n and np.isnan(close[valid_start]):
        valid_start += 1
    
    if valid_start < n:
        kama[valid_start] = close[valid_start]
        
        # Calculate KAMA iteratively
        for i in range(valid_start + 1, n):
            if not np.isnan(close[i]) and not np.isnan(kama[i - 1]):
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


def calculate_kama_slope(kama, lookback=5):
    """Calculate KAMA slope (positive = uptrend, negative = downtrend)"""
    n = len(kama)
    slope = np.zeros(n)
    slope[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(kama[i]) and not np.isnan(kama[i - lookback]):
            slope[i] = kama[i] - kama[i - lookback]
    
    return slope


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators (pre-compute before loop for performance)
    kama = calculate_kama(close, 14)
    kama_slope = calculate_kama_slope(kama, 5)
    atr = calculate_atr(high, low, close, 14)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size (25% of capital - conservative)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(kama_slope[i]) or np.isnan(atr[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Daily trend filter (HTF) - price above 1d HMA = bullish bias
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 4h KAMA trend (slope direction)
        kama_bullish = kama_slope[i] > 0
        kama_bearish = kama_slope[i] < 0
        
        # Price position relative to KAMA (confirmation)
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Determine target signal based on trend alignment
        target_signal = 0.0
        
        # Long entry: Daily bullish + KAMA slope up + Price above KAMA
        if daily_bullish and kama_bullish and price_above_kama:
            target_signal = SIZE
        
        # Short entry: Daily bearish + KAMA slope down + Price below KAMA
        elif daily_bearish and kama_bearish and price_below_kama:
            target_signal = -SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest since entry
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * atr[i]:
                        take_profit_triggered = True
            else:
                # Short position - update lowest since entry
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * atr[i]:
                        take_profit_triggered = True
        
        # Apply signals based on triggers
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
            # Check for trend reversal exit
            trend_reversal = False
            if position_side == 1 and (kama_bearish or not price_above_kama):
                trend_reversal = True
            elif position_side == -1 and (kama_bullish or not price_below_kama):
                trend_reversal = True
            
            if trend_reversal and position_side != 0:
                signals[i] = 0.0
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
                entry_price = 0.0
                profit_target_hit = False
            elif target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals