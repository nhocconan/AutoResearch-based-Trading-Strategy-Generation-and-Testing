#!/usr/bin/env python3
"""
EXPERIMENT #056 - KAMA Adaptive Trend + RSI Pullback + 4h Filter (30m primary)
=====================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better than
EMA/HMA, reducing whipsaws in chop while catching trends quickly. On 30m, KAMA crossovers
with RSI pullback entries should capture trend continuations with better risk/reward than
breakout strategies (Donchian). 4h KAMA filter ensures we trade with the higher timeframe
trend direction. This differs from #047 by using adaptive MA (not Donchian) + pullback
entries (not breakouts) which should reduce false signals in choppy markets.

Key features:
- Primary TF: 30m
- HTF filter: 4h KAMA(21) for trend direction
- Trend: KAMA(10,2,30) fast/slow crossover on 30m
- Entry: RSI(14) pullback (RSI<45 in uptrend, RSI>55 in downtrend)
- Regime: ATR percentile > 40th (avoid ultra-low vol chop)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25 base, discrete levels (0.0, ±0.25, ±0.30)
- Take profit: Reduce to half at 2.5R profit, trail stop at 1.5R

Why this should beat current best (Sharpe=0.490):
- KAMA adapts to volatility = fewer whipsaws than fixed MA
- Pullback entries (not breakouts) = better risk/reward ratio
- 4h trend filter = trade with major trend direction
- Conservative sizing (0.25-0.30) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_pullback_30m_4h_v1"
timeframe = "30m"
leverage = 1.0


def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - moves fast in trends, slow in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(efficiency_period - 1, n):
        signal = abs(close[i] - close[i - efficiency_period + 1])
        noise = 0.0
        for j in range(i - efficiency_period + 2, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[efficiency_period - 1] = close[efficiency_period - 1]
    
    for i in range(efficiency_period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    # Use EMA for smoothing
    gain_smooth = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if loss_smooth[i] == 0:
            rsi[i] = 100.0
        else:
            rs = gain_smooth[i] / loss_smooth[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


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
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    kama_4h = calculate_kama(df_4h['close'].values, efficiency_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Calculate 30m indicators
    kama_fast = calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, efficiency_period=20, fast_period=5, slow_period=50)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    
    # Calculate ATR percentile rank (regime filter)
    atr_pr = calculate_percentile_rank(atr, 100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size
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
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(kama_fast[i]) or 
            np.isnan(kama_slow[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or
            np.isnan(atr_pr[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend direction (HTF filter)
        price_above_4h_kama = close[i] > kama_4h_aligned[i]
        htf_trend = 1 if price_above_4h_kama else -1
        
        # 30m KAMA crossover signal
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # RSI pullback conditions
        rsi_oversold = rsi[i] < 45  # Pullback in uptrend
        rsi_overbought = rsi[i] > 55  # Pullback in downtrend
        
        # Regime filter (avoid ultra-low volatility chop)
        regime_ok = atr_pr[i] > 0.40
        
        # Calculate position size based on ATR regime
        if atr_pr[i] > 0.70:
            position_size = MAX_SIZE
        elif atr_pr[i] > 0.40:
            position_size = BASE_SIZE
        else:
            position_size = MIN_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish + RSI pullback + HTF bullish + regime ok
        if (kama_bullish and rsi_oversold and htf_trend == 1 and regime_ok):
            target_signal = position_size
        
        # Short entry: KAMA bearish + RSI pullback + HTF bearish + regime ok
        elif (kama_bearish and rsi_overbought and htf_trend == -1 and regime_ok):
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
                        
                # Trail stop at 1.5R profit
                if profit_target_hit:
                    trailing_stop = max(trailing_stop, entry_price + 3.75 * entry_atr - 2.5 * atr[i])
                    if close[i] < trailing_stop:
                        stoploss_triggered = True
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
                        
                # Trail stop at 1.5R profit
                if profit_target_hit:
                    trailing_stop = min(trailing_stop, entry_price - 3.75 * entry_atr + 2.5 * atr[i])
                    if close[i] > trailing_stop:
                        stoploss_triggered = True
        
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
                # Exit if KAMA reverses OR HTF alignment breaks
                kama_reversal_long = kama_bearish
                kama_reversal_short = kama_bullish
                hma_alignment_broken = (position_side == 1 and htf_trend == -1) or \
                                       (position_side == -1 and htf_trend == 1)
                
                if kama_reversal_long or kama_reversal_short or hma_alignment_broken:
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