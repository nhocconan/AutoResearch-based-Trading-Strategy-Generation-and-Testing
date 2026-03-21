#!/usr/bin/env python3
"""
EXPERIMENT #064 - HMA Trend + RSI Pullback + Triple HTF Alignment (4h primary)
=====================================================================================
Hypothesis: 4h HMA trend captures medium-term moves, but entering on breakouts causes
whipsaws. Instead, enter on RSI pullbacks WITHIN the trend direction. Triple HTF
alignment (4h price vs 1d HMA vs 1w HMA) ensures we trade with major trend.
Z-score filter avoids entering at extreme extensions.

Key features:
- Primary TF: 4h (as required by experiment)
- HTF filters: 1d HMA(50) + 1w HMA(50) for triple alignment
- Trend: 4h HMA(21) vs HMA(48) crossover
- Entry: RSI(14) pullback to 40-50 (long) or 50-60 (short) within trend
- Regime: Z-score(20) < 2.0 (avoid extreme extensions)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25 base, discrete levels (0.0, ±0.25, ±0.30)
- Take profit: Reduce to half at 2R profit, trail stop at 1R

Why this should beat current best (Sharpe=0.490):
- Pullback entries have better risk/reward than breakouts
- 4h timeframe reduces noise vs 15m/30m/1h strategies that failed
- Triple HTF alignment filters counter-trend trades
- Z-score avoids chasing extended moves
- Conservative sizing (0.25-0.30) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_pullback_triplehtf_4h_1d_1w_v1"
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
    """Calculate RSI"""
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std
    return zscore.values


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
    hma_48 = calculate_hma(close, 48)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size with strong confirmation
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
            np.isnan(hma_21[i]) or np.isnan(hma_48[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(zscore[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Triple HTF trend alignment
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        
        # 4h trend direction (HMA crossover)
        hma_21_above_48 = hma_21[i] > hma_48[i]
        
        # 1d and 1w trend direction
        daily_trend = 1 if price_above_1d_hma else -1
        weekly_trend = 1 if price_above_1w_hma else -1
        hma_trend = 1 if hma_21_above_48 else -1
        
        # Z-score regime filter (avoid extreme extensions)
        zscore_normal = abs(zscore[i]) < 2.0
        
        # RSI pullback conditions
        # Long: RSI pulled back to 40-50 in uptrend (buying the dip)
        rsi_pullback_long = 40 <= rsi[i] <= 55
        # Short: RSI pulled back to 45-60 in downtrend (selling the rip)
        rsi_pullback_short = 45 <= rsi[i] <= 60
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: 4h HMA bullish + Triple HTF alignment + RSI pullback + Z-score normal
        if (hma_trend == 1 and daily_trend == 1 and weekly_trend == 1 and
            rsi_pullback_long and zscore_normal):
            target_signal = BASE_SIZE
        
        # Short entry: 4h HMA bearish + Triple HTF alignment + RSI pullback + Z-score normal
        elif (hma_trend == -1 and daily_trend == -1 and weekly_trend == -1 and
              rsi_pullback_short and zscore_normal):
            target_signal = -BASE_SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * entry_atr:  # 2R = 4*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * entry_atr:  # 2R profit
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
                # Exit if HMA crossover reverses OR HTF alignment breaks
                hma_reversal_long = not hma_21_above_48
                hma_reversal_short = hma_21_above_48
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
                    signals[i] = BASE_SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals