#!/usr/bin/env python3
"""
EXPERIMENT #066 - HMA Trend + RSI Pullback + Weekly Filter (1d primary)
========================================================================
Hypothesis: Daily timeframe captures major trend moves with less noise than intraday.
RSI pullbacks in established trends (HMA slope + weekly alignment) provide high-probability
entries with favorable risk/reward. Weekly HMA filter ensures we trade with the major trend.
This differs from 12h strategies by using cleaner daily signals with weekly confirmation.

Key features:
- Primary TF: 1d (daily candles - less noise, clearer trends)
- HTF filter: 1w HMA(50) for major trend direction
- Trend: HMA(21) vs HMA(50) crossover + slope confirmation
- Entry: RSI(14) pullback to 40-60 zone in established trend
- Regime: Z-score(20) filter to avoid extreme overbought/oversold
- Stoploss: 2.5*ATR(14) trailing stop
- Position sizing: 0.25-0.30 discrete levels

Why this should beat current best (Sharpe=0.490):
- Daily timeframe has cleaner signals than 12h (less whipsaw)
- Weekly filter prevents counter-trend trades in major corrections
- RSI pullback entries have better risk/reward than breakouts
- Conservative sizing (0.25-0.30) controls drawdown in crypto volatility
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_pullback_weekly_1d_1w_v3"
timeframe = "1d"
leverage = 1.0


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
    delta = np.zeros(n)
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rsi = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std
    return zscore.values


def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope (rate of change over lookback periods)"""
    n = len(hma_values)
    slope = np.zeros(n)
    for i in range(lookback, n):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback]
    return slope


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    hma_21_slope = calculate_hma_slope(hma_21, 5)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.32   # Max position size
    MIN_SIZE = 0.22   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(zscore[i]) or
            np.isnan(hma_21_slope[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter (major trend direction)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily trend confirmation (HMA 21 vs 50)
        daily_bullish = hma_21[i] > hma_50[i]
        daily_bearish = hma_21[i] < hma_50[i]
        
        # HMA slope confirmation (trend momentum)
        hma_slope_positive = hma_21_slope[i] > 0.001  # >0.1% per 5 days
        hma_slope_negative = hma_21_slope[i] < -0.001
        
        # RSI pullback zones (not extreme)
        rsi_pullback_long = 35 < rsi[i] < 55  # Pullback in uptrend
        rsi_pullback_short = 45 < rsi[i] < 65  # Pullback in downtrend
        
        # Z-score regime filter (avoid extremes)
        zscore_normal = abs(zscore[i]) < 2.0  # Not extreme deviation
        
        # Calculate position size based on trend strength
        trend_strength = abs(hma_21_slope[i]) * 1000  # Scale for readability
        size_multiplier = min(1.0 + trend_strength, 1.15)  # Max 1.15x
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * size_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Weekly bullish + Daily bullish + HMA slope up + RSI pullback + Z-score normal
        if (weekly_bullish and daily_bullish and hma_slope_positive and 
            rsi_pullback_long and zscore_normal):
            target_signal = position_size
        
        # Short entry: Weekly bearish + Daily bearish + HMA slope down + RSI pullback + Z-score normal
        elif (weekly_bearish and daily_bearish and hma_slope_negative and 
              rsi_pullback_short and zscore_normal):
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
                
                # Check take profit (2R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2R = 5*ATR
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
                    if close[i] <= entry_price - 5.0 * entry_atr:  # 2R profit
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
                # Exit if HMA crossover reverses OR weekly trend breaks
                hma_reversal_long = hma_21[i] < hma_50[i]
                hma_reversal_short = hma_21[i] > hma_50[i]
                weekly_trend_broken = (position_side == 1 and weekly_bearish) or \
                                      (position_side == -1 and weekly_bullish)
                
                if hma_reversal_long or hma_reversal_short or weekly_trend_broken:
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