#!/usr/bin/env python3
"""
EXPERIMENT #057 - HMA Trend + RSI Pullback + Bollinger Regime (1h primary)
=====================================================================================
Hypothesis: 1h HMA trend following with RSI pullback entries captures trend continuations
while avoiding chasing breakouts. Bollinger Band width filters out choppy regimes.
4h and 1d HMA alignment ensures we trade with higher timeframe trend direction.

Key features:
- Primary TF: 1h
- HTF filters: 4h HMA(21) + 1d HMA(50) for trend alignment
- Trend: HMA(16) vs HMA(48) crossover on 1h
- Entry: RSI(14) pullback to 40-60 zone in trending market
- Regime: Bollinger Band width > 40th percentile (avoid squeeze/chop)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete, scaled by regime strength
- Take profit: Reduce to half at 2R profit, trail stop at 1R

Why this should beat current best (Sharpe=0.490):
- RSI pullbacks enter on dips rather than breakouts (better risk/reward)
- Bollinger regime filter avoids 50%+ of choppy periods
- Triple HMA alignment (1h/4h/1d) ensures strong trend confirmation
- Conservative sizing (0.25-0.30) with discrete levels reduces fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_bb_regime_1h_4h_1d_v1"
timeframe = "1h"
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
    delta = np.diff(close, prepend=close[0])
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma  # Normalized bandwidth
    return upper, lower, sma, bandwidth


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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1h indicators
    hma_1h_fast = calculate_hma(close, 16)
    hma_1h_slow = calculate_hma(close, 48)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    
    # Calculate Bollinger Band width percentile (regime filter)
    bb_pr = calculate_percentile_rank(bb_bandwidth, 100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size with strong regime
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
            np.isnan(hma_1h_fast[i]) or np.isnan(hma_1h_slow[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(bb_bandwidth[i]) or
            np.isnan(bb_pr[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Triple HMA trend alignment
        hma_1h_bullish = hma_1h_fast[i] > hma_1h_slow[i]
        hma_1h_bearish = hma_1h_fast[i] < hma_1h_slow[i]
        
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # HTF trend direction
        htf_trend_4h = 1 if price_above_4h_hma else -1
        htf_trend_1d = 1 if price_above_1d_hma else -1
        
        # Bollinger Band regime filter (avoid chop/squeeze)
        bb_regime_ok = bb_pr[i] > 0.40  # Only trade when bandwidth > 40th percentile
        
        # RSI pullback conditions (enter on dips in uptrend, rallies in downtrend)
        rsi_pullback_long = 35 <= rsi[i] <= 55  # Pullback zone for longs
        rsi_pullback_short = 45 <= rsi[i] <= 65  # Rally zone for shorts
        
        # HMA crossover confirmation (fast crossing above/below slow)
        hma_cross_long = hma_1h_fast[i] > hma_1h_slow[i] and hma_1h_fast[i-1] <= hma_1h_slow[i-1]
        hma_cross_short = hma_1h_fast[i] < hma_1h_slow[i] and hma_1h_fast[i-1] >= hma_1h_slow[i-1]
        
        # Calculate position size based on regime strength
        regime_multiplier = min(1.0 + (bb_pr[i] - 0.40) * 0.5, 1.25)  # Max 1.25x
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * regime_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: HMA bullish + RSI pullback + HTF alignment + BB regime OK
        if (hma_1h_bullish and rsi_pullback_long and 
            htf_trend_4h == 1 and htf_trend_1d == 1 and bb_regime_ok):
            target_signal = position_size
        
        # Short entry: HMA bearish + RSI pullback + HTF alignment + BB regime OK
        elif (hma_1h_bearish and rsi_pullback_short and 
              htf_trend_4h == -1 and htf_trend_1d == -1 and bb_regime_ok):
            target_signal = -position_size
        
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
                # Exit if HMA crosses against position OR HTF alignment breaks
                hma_reversal_long = hma_1h_fast[i] < hma_1h_slow[i]
                hma_reversal_short = hma_1h_fast[i] > hma_1h_slow[i]
                hma_alignment_broken = (position_side == 1 and htf_trend_4h == -1) or \
                                       (position_side == -1 and htf_trend_4h == 1)
                
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
                    signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals