#!/usr/bin/env python3
"""
EXPERIMENT #010 - KAMA Adaptive Trend + ADX Strength + RSI Timing (4h primary, 1d HTF)
======================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better
than fixed EMAs/HMAs. During trending markets, KAMA follows price closely; during
chop, it flattens to avoid whipsaws. Combined with ADX(14) > 25 for trend strength
confirmation and 1d KAMA(50) for major trend alignment. RSI(14) filters entry timing
to avoid buying/selling at extremes.

Key features:
- Primary TF: 4h (as required for this experiment)
- HTF filter: 1d KAMA(50) for major trend direction
- Trend: KAMA(10) vs KAMA(40) crossover on 4h
- Strength: ADX(14) > 25 ensures we only trade in trending markets
- Entry timing: RSI(14) between 35-65 (avoid extremes)
- Stoploss: 2.5*ATR(14) trailing stop
- Position sizing: 0.30 discrete levels (30% of capital max)
- Take profit: Reduce to half at 2R profit (5*ATR from entry)

Why this differs from failed attempts:
- KAMA adapts to volatility (unlike fixed HMA/DEMA/EMA in #001-#009)
- ADX filter avoids trading in chop (missing in #002, #007, #008)
- Proper HTF alignment using mtf_data helper ( Rule 1-3 compliance)
- Conservative position sizing at 0.30 max ( Rule 4 compliance)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_adx_rsi_4h_1d_v1"
timeframe = "4h"
leverage = 1.0


def calculate_kama(close, fast_period=2, slow_period=30, smoothing_period=10):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market volatility using Efficiency Ratio (ER)
    
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < slow_period + smoothing_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(slow_period, n):
        signal = abs(close[i] - close[i - slow_period])
        noise = 0.0
        for j in range(i - slow_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.zeros(n)
    sc[:] = np.nan
    
    for i in range(slow_period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA with SMA of first slow_period bars
    kama[slow_period - 1] = np.nanmean(close[:slow_period])
    
    # Calculate KAMA
    for i in range(slow_period, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1] if i > 0 else close[0]
    
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


def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 indicates strong trend
    """
    n = len(close)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Calculate TR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    
    # Smooth +DM, -DM, and TR using Wilder's method (EMA with span=period)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    tr_s = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_s[i] > 0:
            plus_di[i] = 100 * plus_dm_s[i] / tr_s[i]
            minus_di[i] = 100 * minus_dm_s[i] / tr_s[i]
    
    # Calculate DX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    kama_1d = calculate_kama(df_1d['close'].values, fast_period=2, slow_period=50, smoothing_period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)  # auto shift(1) for completed bars
    
    # Calculate 4h indicators
    kama_fast = calculate_kama(close, fast_period=2, slow_period=10, smoothing_period=10)
    kama_slow = calculate_kama(close, fast_period=2, slow_period=40, smoothing_period=10)
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.30  # Base position size (30% of capital - Rule 4)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 1.0  # ATR at entry for R calculation
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(kama_fast[i]) or 
            np.isnan(kama_slow[i]) or np.isnan(atr[i]) or np.isnan(adx[i]) or 
            np.isnan(rsi[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Daily trend filter (HTF) - price above/below 1d KAMA(50)
        daily_trend = 1 if close[i] > kama_1d_aligned[i] else -1
        
        # 4h KAMA crossover trend
        kama_trend = 0
        if kama_fast[i] > kama_slow[i]:
            kama_trend = 1  # Bullish
        elif kama_fast[i] < kama_slow[i]:
            kama_trend = -1  # Bearish
        
        # ADX trend strength filter (only trade when ADX > 25)
        trend_strong = adx[i] > 25.0
        
        # RSI entry timing filter (avoid extremes)
        rsi_valid_long = 35 <= rsi[i] <= 65
        rsi_valid_short = 35 <= rsi[i] <= 65
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish + Daily trend bullish + ADX strong + RSI valid
        if kama_trend == 1 and daily_trend == 1 and trend_strong and rsi_valid_long:
            target_signal = SIZE
        
        # Short entry: KAMA bearish + Daily trend bearish + ADX strong + RSI valid
        elif kama_trend == -1 and daily_trend == -1 and trend_strong and rsi_valid_short:
            target_signal = -SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        trend_reversal_exit = False
        
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
                
                # Check trend reversal exit
                if kama_trend == -1:
                    trend_reversal_exit = True
                    
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
                
                # Check trend reversal exit
                if kama_trend == 1:
                    trend_reversal_exit = True
        
        # Apply signals based on conditions
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 1.0
            profit_target_hit = False
            
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
            
        elif trend_reversal_exit:
            # Exit on trend reversal
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 1.0
            profit_target_hit = False
            
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
                # Maintain existing position
                signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals