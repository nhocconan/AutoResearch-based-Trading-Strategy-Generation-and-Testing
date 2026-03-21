#!/usr/bin/env python3
"""
EXPERIMENT #067 - Supertrend + Volume + BB Regime + 4h HTF Filter (15m primary)
=====================================================================================
Hypothesis: 15m Supertrend captures intraday trends but generates many false signals in chop.
Adding volume confirmation (volume > 1.5x 20-period avg) filters low-conviction moves.
Bollinger Band width percentile detects regime - only trade when BW > 40th percentile (trending).
4h HMA(50) provides major trend filter - only trade in direction of HTF trend.

Key features:
- Primary TF: 15m
- HTF filter: 4h HMA(50) for major trend direction
- Trend: Supertrend(10, 3.0) for entry timing
- Volume: volume > 1.5x 20-period average (confirms breakout)
- Regime: Bollinger Band width > 40th percentile (avoid squeeze/chop)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25 base, 0.30 max with strong volume
- Take profit: Reduce to half at 2.5R profit

Why this should beat current best (Sharpe=0.490):
- 15m captures more intraday moves than 12h
- Volume filter removes 40%+ of false breakouts
- BB width regime filter avoids chop periods
- 4h HTF ensures we trade with major trend
- Conservative sizing (0.25-0.30) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_vol_bbregime_4hhtf_15m_v1"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    supertrend_direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = close[i]
            supertrend_direction[i] = 0
            continue
        
        upper_band = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band = (high[i] + low[i]) / 2 - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            supertrend_direction[i] = -1
        else:
            # Bullish condition
            if close[i - 1] > supertrend[i - 1]:
                supertrend[i] = min(lower_band, supertrend[i - 1])
                if close[i] < supertrend[i]:
                    supertrend[i] = upper_band
                    supertrend_direction[i] = -1
                else:
                    supertrend_direction[i] = 1
            # Bearish condition
            else:
                supertrend[i] = max(upper_band, supertrend[i - 1])
                if close[i] > supertrend[i]:
                    supertrend[i] = lower_band
                    supertrend_direction[i] = 1
                else:
                    supertrend_direction[i] = -1
    
    return supertrend, supertrend_direction


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    return upper, lower, bandwidth


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


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    supertrend, supertrend_dir = calculate_supertrend(high, low, close, 10, 3.0)
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    
    # Volume moving average
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # BB width percentile rank (regime filter)
    bb_width_pr = calculate_percentile_rank(bb_bandwidth, 100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size with strong volume
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
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(supertrend[i]) or
            np.isnan(atr[i]) or np.isnan(bb_bandwidth[i]) or
            np.isnan(bb_width_pr[i]) or np.isnan(volume_sma[i]) or
            atr[i] == 0 or volume_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HTF trend direction
        hma_4h_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # Supertrend direction
        st_direction = supertrend_dir[i]
        
        # Volume confirmation (volume > 1.5x 20-period average)
        volume_ratio = volume[i] / volume_sma[i] if volume_sma[i] > 0 else 0
        volume_confirmed = volume_ratio > 1.5
        
        # BB width regime filter (only trade when bandwidth > 40th percentile)
        bb_regime_ok = bb_width_pr[i] > 0.40
        
        # Calculate position size based on volume strength
        if volume_ratio > 2.0:
            position_size = MAX_SIZE
        else:
            position_size = BASE_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Supertrend bullish + 4h trend up + volume confirmed + BB regime OK
        if (st_direction == 1 and hma_4h_trend == 1 and 
            volume_confirmed and bb_regime_ok):
            target_signal = position_size
        
        # Short entry: Supertrend bearish + 4h trend down + volume confirmed + BB regime OK
        elif (st_direction == -1 and hma_4h_trend == -1 and 
              volume_confirmed and bb_regime_ok):
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
                # Exit if Supertrend reverses OR 4h HTF alignment breaks
                st_reversal = (position_side == 1 and st_direction == -1) or \
                              (position_side == -1 and st_direction == 1)
                hma_alignment_broken = (position_side == 1 and hma_4h_trend == -1) or \
                                       (position_side == -1 and hma_4h_trend == 1)
                
                if st_reversal or hma_alignment_broken:
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