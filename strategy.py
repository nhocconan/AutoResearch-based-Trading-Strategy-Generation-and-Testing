#!/usr/bin/env python3
"""
EXPERIMENT #001 - Multi-Timeframe EMA Trend + RSI Pullback Strategy
====================================================================
Hypothesis: Daily EMA(21/55) trend filter combined with 4h RSI(14) pullback 
entries will capture major crypto trends while avoiding counter-trend whipsaws.
Daily trend determines bias, 4h RSI identifies optimal entry points within trend.

Why this should beat baseline:
- Daily trend filter eliminates 50%+ of false signals
- RSI pullbacks enter at better prices (mean reversion within trend)
- 4h timeframe = cleaner signals than 1h, more trades than 1d
- ATR-based stoploss adapts to volatility regimes
- Discrete position sizing minimizes fee churn

Setup: Primary=4h, HTF=1d (Experiment #2 from list)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_ema_rsi_pullback_4h_v2"
timeframe = "4h"
leverage = 1.0


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI with proper min_periods handling."""
    n = len(close)
    rsi = np.zeros(n)
    
    if n < period + 1:
        return rsi
    
    # Calculate price changes
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)  # Pad to match length
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Calculate average gains/losses using EMA-style smoothing
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    # First SMA for initial values
    avg_gain[period] = np.mean(gains[1:period+1])
    avg_loss[period] = np.mean(losses[1:period+1])
    
    # EMA smoothing for rest
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
    
    # Calculate RS and RSI
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100  # When no losses, RSI = 100
    
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100
    
    return rsi


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                  period: int = 14) -> np.ndarray:
    """Calculate ATR with proper min_periods handling."""
    n = len(close)
    tr = np.zeros(n)
    atr = np.zeros(n)
    
    if n < 2:
        return atr
    
    # True Range
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # First SMA
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        
        # EMA smoothing for rest
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_ema(series: np.ndarray, span: int) -> np.ndarray:
    """Calculate EMA with proper min_periods handling."""
    n = len(series)
    ema = np.zeros(n)
    
    if n < span:
        return ema
    
    # First SMA
    ema[span-1] = np.mean(series[:span])
    
    # EMA calculation
    multiplier = 2 / (span + 1)
    for i in range(span, n):
        ema[i] = (series[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Generate trading signals using multi-timeframe approach.
    
    Logic:
    1. Daily EMA(21) vs EMA(55) determines trend bias
    2. 4h RSI(14) identifies pullback entries within trend
    3. ATR(14) for stoploss and position sizing
    4. Discrete signal levels to minimize fee churn
    """
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # =====================================================
    # LOAD HTF DATA ONCE BEFORE LOOP (CRITICAL RULE #1)
    # =====================================================
    df_1d = get_htf_data(prices, '1d')  # Load daily data ONCE
    
    # Calculate daily EMAs
    daily_close = df_1d['close'].values
    daily_ema21 = calculate_ema(daily_close, 21)
    daily_ema55 = calculate_ema(daily_close, 55)
    
    # Daily trend: 1 = bullish (ema21 > ema55), -1 = bearish, 0 = neutral
    daily_trend_raw = np.zeros(len(daily_close))
    for i in range(55, len(daily_close)):
        if daily_ema21[i] > daily_ema55[i] * 1.001:  # 0.1% buffer
            daily_trend_raw[i] = 1
        elif daily_ema21[i] < daily_ema55[i] * 0.999:
            daily_trend_raw[i] = -1
    
    # Align daily trend to 4h timeframe (auto shift(1) for completed bars)
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_raw)
    
    # =====================================================
    # CALCULATE 4H INDICATORS
    # =====================================================
    rsi_14 = calculate_rsi(close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # =====================================================
    # GENERATE SIGNALS
    # =====================================================
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (Rule #4)
    BASE_SIZE = 0.30  # 30% of capital
    HALF_SIZE = 0.15  # 15% for take profit
    
    # Entry thresholds
    RSI_LONG_ENTRY = 38   # Buy pullback in uptrend
    RSI_SHORT_ENTRY = 62  # Sell pullback in downtrend
    RSI_EXIT = 50         # Neutral exit
    
    # Stoploss multiplier
    STOPLOSS_MULT = 2.5   # 2.5 * ATR
    
    # Track position state
    position_side = 0     # 0=flat, 1=long, -1=short
    entry_price = 0.0
    stop_loss = 0.0
    take_profit_level = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period for all indicators
    warmup = max(55, 100)
    
    for i in range(warmup, n):
        # Skip if ATR is zero or NaN
        if atr_14[i] <= 0 or np.isnan(atr_14[i]):
            signals[i] = 0.0
            continue
        
        # Get aligned daily trend (already shifted for completed bars)
        trend = daily_trend_aligned[i]
        
        # Current RSI
        rsi = rsi_14[i]
        
        # =================================================
        # ENTRY LOGIC
        # =================================================
        if position_side == 0:
            # Long entry: Daily bullish + RSI pullback
            if trend == 1 and rsi < RSI_LONG_ENTRY:
                position_side = 1
                entry_price = close[i]
                highest_since_entry = close[i]
                stop_loss = entry_price - STOPLOSS_MULT * atr_14[i]
                take_profit_level = entry_price + 2 * STOPLOSS_MULT * atr_14[i]
                signals[i] = BASE_SIZE
            
            # Short entry: Daily bearish + RSI pullback
            elif trend == -1 and rsi > RSI_SHORT_ENTRY:
                position_side = -1
                entry_price = close[i]
                lowest_since_entry = close[i]
                stop_loss = entry_price + STOPLOSS_MULT * atr_14[i]
                take_profit_level = entry_price - 2 * STOPLOSS_MULT * atr_14[i]
                signals[i] = -BASE_SIZE
        
        # =================================================
        # EXIT / POSITION MANAGEMENT LOGIC
        # =================================================
        elif position_side == 1:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, close[i])
            
            # Trail stop loss at 1R profit
            if close[i] >= entry_price + STOPLOSS_MULT * atr_14[i]:
                trailing_stop = highest_since_entry - STOPLOSS_MULT * atr_14[i]
                stop_loss = max(stop_loss, trailing_stop)
            
            # Take profit: reduce to half position at 2R
            if close[i] >= take_profit_level and signals[i-1] == BASE_SIZE:
                signals[i] = HALF_SIZE
            # Close remaining at trend reversal or stoploss
            elif close[i] <= stop_loss or trend == -1 or rsi > RSI_EXIT:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = BASE_SIZE
        
        elif position_side == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, close[i])
            
            # Trail stop loss at 1R profit
            if close[i] <= entry_price - STOPLOSS_MULT * atr_14[i]:
                trailing_stop = lowest_since_entry + STOPLOSS_MULT * atr_14[i]
                stop_loss = min(stop_loss, trailing_stop)
            
            # Take profit: reduce to half position at 2R
            if close[i] <= take_profit_level and signals[i-1] == -BASE_SIZE:
                signals[i] = -HALF_SIZE
            # Close remaining at trend reversal or stoploss
            elif close[i] >= stop_loss or trend == 1 or rsi < RSI_EXIT:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = -BASE_SIZE
    
    return signals