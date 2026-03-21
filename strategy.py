#!/usr/bin/env python3
"""
EXPERIMENT #014 - HMA Crossover + RSI Pullback + Z-Score Filter (30m primary, 4h/1d HTF)
========================================================================================
Hypothesis: 30m HMA(8/21) crossover provides timely entry signals, but only when
aligned with 4h HMA(21) trend and 1d HMA(50) major trend. RSI(14) in neutral zone
(45-55) ensures we enter on pullbacks, not extremes. Z-score(20) filter avoids
entering when price is extended (>2 std dev). This differs from failed strategies
by using HMA (faster, less lag) instead of Supertrend/KAMA, and combining 3 signal
types for confirmation.

Key features:
- Primary TF: 30m (as required for this experiment)
- HTF filter 1: 4h HMA(21) for intermediate trend
- HTF filter 2: 1d HMA(50) for major trend alignment
- Entry: 30m HMA(8) crosses HMA(21) in trend direction
- Timing: RSI(14) in 45-55 zone (pullback, not extreme)
- Filter: Z-score(20) < 2.0 (price not extended)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_zscore_30m_4h_1d_v1"
timeframe = "30m"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average - faster response than EMA"""
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
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / (std + 1e-10)
    return zscore.values


def calculate_hma_crossover(hma_fast, hma_slow):
    """Detect HMA crossover signals"""
    n = len(hma_fast)
    crossover = np.zeros(n)
    
    for i in range(1, n):
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            continue
        # Bullish crossover: fast crosses above slow
        if hma_fast[i-1] <= hma_slow[i-1] and hma_fast[i] > hma_slow[i]:
            crossover[i] = 1
        # Bearish crossover: fast crosses below slow
        elif hma_fast[i-1] >= hma_slow[i-1] and hma_fast[i] < hma_slow[i]:
            crossover[i] = -1
    
    return crossover


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA(21) for intermediate trend
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA(50) for major trend
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    hma_8 = calculate_hma(close, 8)
    hma_21 = calculate_hma(close, 21)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    
    # Detect HMA crossovers
    hma_cross = calculate_hma_crossover(hma_8, hma_21)
    
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
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]) or
            np.isnan(hma_8[i]) or np.isnan(hma_21[i]) or np.isnan(atr[i]) or
            np.isnan(rsi[i]) or np.isnan(zscore[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # HTF Trend filters
        htf_4h_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        htf_1d_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        
        # 30m HMA crossover signal
        cross_signal = hma_cross[i]
        
        # RSI pullback zone (45-55 = neutral, not extreme)
        rsi_neutral = 45 <= rsi[i] <= 55
        
        # Z-score filter (avoid extended moves > 2 std dev)
        zscore_valid = abs(zscore[i]) < 2.0
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: HTF trends bullish + HMA bullish cross + RSI neutral + Z-score valid
        if (htf_4h_trend == 1 and htf_1d_trend == 1 and 
            cross_signal == 1 and rsi_neutral and zscore_valid):
            target_signal = SIZE
        
        # Short entry: HTF trends bearish + HMA bearish cross + RSI neutral + Z-score valid
        elif (htf_4h_trend == -1 and htf_1d_trend == -1 and 
              cross_signal == -1 and rsi_neutral and zscore_valid):
            target_signal = -SIZE
        
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
                
                # Check take profit (2R from entry, where R = 2*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * atr[i]:  # 2R = 4*ATR
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
                    if close[i] <= entry_price - 4.0 * atr[i]:  # 2R profit
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
                if position_side == 1 and htf_4h_trend == -1:
                    # 4h trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and htf_4h_trend == 1:
                    # 4h trend reversed, exit short
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