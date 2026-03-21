#!/usr/bin/env python3
"""
EXPERIMENT #033 - DEMA Crossover + HTF Trend + ADX Strength + Z-Score Timing (1h primary, 4h HTF)
================================================================================================
Hypothesis: DEMA (Double EMA) reacts faster than HMA for entry timing while maintaining
trend accuracy. Combined with 4h HMA for major trend direction, ADX(14)>25 for trend
strength filter, and Z-score(20) for pullback entry timing within the trend. This differs
from previous HMA strategies by using DEMA for faster signals and adding ADX strength filter
to avoid choppy markets. Z-score ensures we enter on pullbacks, not breakouts.

Key features:
- Primary TF: 1h (this experiment)
- HTF filter: 4h HMA(21) for major trend direction
- Trend: DEMA(8) vs DEMA(21) crossover on 1h
- Strength: ADX(14) > 25 (only trade strong trends)
- Timing: Z-score(20) between -1.5 and 1.5 for pullback entries
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit, trail stop at 1R

Why this should work:
- DEMA reduces lag vs EMA while being smoother than raw price
- 4h HMA filter prevents counter-trend trades (proven in best strategy)
- ADX filter avoids choppy sideways markets (major cause of losses)
- Z-score timing ensures we buy dips in uptrends, sell rallies in downtrends
- Conservative sizing (0.25-0.30) limits drawdown during adverse moves
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "dema_adx_zscore_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_dema(close, period):
    """Calculate Double Exponential Moving Average"""
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    dema = 2 * ema1 - ema2
    return dema.values


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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    # Calculate True Range, +DM, -DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM using Wilder's method (EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate DI+ and DI-
    plus_di_pct = np.zeros(n)
    minus_di_pct = np.zeros(n)
    
    for i in range(period - 1, n):
        if atr[i] > 0:
            plus_di_pct[i] = 100 * plus_di[i] / atr[i]
            minus_di_pct[i] = 100 * minus_di[i] / atr[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period - 1, n):
        di_sum = plus_di_pct[i] + minus_di_pct[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di_pct[i] - minus_di_pct[i]) / di_sum
    
    # Smooth DX to get ADX
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized price deviation from mean)"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / (std + 1e-10)
    return zscore.values


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    dema_fast = calculate_dema(close, 8)
    dema_slow = calculate_dema(close, 21)
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    
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
    atr_at_entry = 0.0
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(dema_fast[i]) or 
            np.isnan(dema_slow[i]) or np.isnan(atr[i]) or np.isnan(adx[i]) or 
            np.isnan(zscore[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HTF trend filter (major trend direction)
        htf_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # 1h DEMA crossover signal
        dema_signal = 0
        if dema_fast[i] > dema_slow[i]:
            dema_signal = 1  # Bullish crossover
        elif dema_fast[i] < dema_slow[i]:
            dema_signal = -1  # Bearish crossover
        
        # ADX strength filter (only trade when ADX > 25, strong trend)
        trend_strong = adx[i] > 25.0
        
        # Z-score timing (enter on pullbacks, not breakouts)
        # For longs: Z-score between -1.5 and 0.5 (pullback within uptrend)
        # For shorts: Z-score between -0.5 and 1.5 (rally within downtrend)
        zscore_long = -1.5 <= zscore[i] <= 0.5
        zscore_short = -0.5 <= zscore[i] <= 1.5
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: HTF bullish + DEMA bullish + ADX strong + Z-score pullback
        if htf_trend == 1 and dema_signal == 1 and trend_strong and zscore_long:
            target_signal = SIZE
        
        # Short entry: HTF bearish + DEMA bearish + ADX strong + Z-score rally
        elif htf_trend == -1 and dema_signal == -1 and trend_strong and zscore_short:
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
                
                # Check take profit (2R from entry, where R = 2.0*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * atr_at_entry:  # 2R = 4*ATR
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
                    if close[i] <= entry_price - 4.0 * atr_at_entry:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
            atr_at_entry = 0.0
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
                atr_at_entry = atr[i]
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
                    atr_at_entry = 0.0
                elif position_side == -1 and dema_signal == 1:
                    # DEMA crossed bullish, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                    atr_at_entry = 0.0
                else:
                    # Maintain position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals