#!/usr/bin/env python3
"""
EXPERIMENT #031 - MACD Histogram + Bollinger Squeeze + 4h HMA Trend (15m primary)
=====================================================================================
Hypothesis: MACD histogram captures momentum shifts earlier than Supertrend.
Bollinger Band squeeze identifies low-volatility regimes before breakouts.
Combining MACD momentum + BB squeeze + 4h HMA trend filter should capture
breakouts with better timing than pure trend-following approaches.

Key features:
- Primary TF: 15m
- HTF filter: 4h HMA(21) for major trend direction
- Momentum: MACD(12,26,9) histogram for entry timing
- Regime: Bollinger Band Width percentile for squeeze detection
- Entry: MACD histogram cross + BB squeeze release + HTF trend alignment
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this should beat previous attempts:
- MACD histogram leads price (momentum before trend)
- BB squeeze filters choppy periods (only trade when volatility expands)
- 15m captures more opportunities than 1h/4h strategies
- Conservative sizing controls drawdown during crypto crashes
- Simpler entry logic than Supertrend+RSI+ADX (which had too many filters)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "macd_bbqueeze_4hhma_15m_v1"
timeframe = "15m"
leverage = 1.0


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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_s.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper.values, lower.values, sma.values, std.values


def calculate_bb_width(upper, lower, sma):
    """Calculate Bollinger Band Width (normalized)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        bb_width = (upper - lower) / sma
    bb_width = np.nan_to_num(bb_width, nan=0.0)
    return bb_width


def calculate_zscore(values, period=20):
    """Calculate Z-score of values"""
    values_s = pd.Series(values)
    mean = values_s.rolling(window=period, min_periods=period).mean()
    std = values_s.rolling(window=period, min_periods=period).std()
    zscore = (values_s - mean) / std
    return zscore.values


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


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
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend filter
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    macd_line, signal_line, histogram = calculate_macd(close, fast=12, slow=26, signal=9)
    bb_upper, bb_lower, bb_sma, bb_std = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_sma)
    bb_width_zscore = calculate_zscore(bb_width, period=100)  # Z-score of BB width
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    # Track MACD histogram for crossover detection
    prev_histogram = 0.0
    
    min_period = 120  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(histogram[i]) or
            np.isnan(bb_width_zscore[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            prev_histogram = histogram[i] if not np.isnan(histogram[i]) else prev_histogram
            continue
        
        # 4h HMA trend filter
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_trend = 1 if price_above_4h_hma else -1
        
        # MACD histogram momentum
        hist_positive = histogram[i] > 0
        hist_negative = histogram[i] < 0
        
        # MACD histogram crossover detection
        hist_cross_up = (prev_histogram < 0) and (histogram[i] > 0)
        hist_cross_down = (prev_histogram > 0) and (histogram[i] < 0)
        
        # Bollinger Band squeeze detection (BB width in bottom 30% = squeeze)
        bb_squeeze = bb_width_zscore[i] < -0.5  # Below average width
        
        # BB squeeze release (width expanding from squeeze)
        bb_release_long = bb_squeeze and (histogram[i] > prev_histogram) and hist_positive
        bb_release_short = bb_squeeze and (histogram[i] < prev_histogram) and hist_negative
        
        # RSI filter (avoid extreme overbought/oversold for entries)
        rsi_ok_long = rsi[i] < 70  # Not overbought
        rsi_ok_short = rsi[i] > 30  # Not oversold
        
        # Calculate position size
        position_size = BASE_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: 4h HMA bullish + MACD hist cross up OR BB squeeze release + RSI ok
        if (hma_trend == 1 and rsi_ok_long and
            (hist_cross_up or bb_release_long)):
            target_signal = position_size
        
        # Short entry: 4h HMA bearish + MACD hist cross down OR BB squeeze release + RSI ok
        elif (hma_trend == -1 and rsi_ok_short and
              (hist_cross_down or bb_release_short)):
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
                # Exit if 4h HMA alignment breaks OR MACD histogram strongly reverses
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
                macd_strong_reversal = False
                if position_side == 1 and histogram[i] < -histogram[i-10] if i >= 10 else False:
                    macd_strong_reversal = True
                elif position_side == -1 and histogram[i] > -histogram[i-10] if i >= 10 else False:
                    macd_strong_reversal = True
                
                if hma_alignment_broken:
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
        
        # Update previous histogram for next iteration
        prev_histogram = histogram[i]
    
    return signals