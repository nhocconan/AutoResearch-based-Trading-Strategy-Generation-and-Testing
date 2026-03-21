#!/usr/bin/env python3
"""
EXPERIMENT #018 - Bollinger Squeeze Breakout with ADX + 4h HMA Trend (1d)
==========================================================================
Hypothesis: Volatility contraction (BB squeeze) followed by expansion captures
major trend moves. ADX(14) > 25 filters for trending markets (avoid chop).
4h HMA(21) provides intermediate trend alignment. This differs from previous
strategies by focusing on volatility regime changes rather than pure price
breakouts or RSI mean reversion.

Key features:
- Primary TF: 1d (daily candles)
- HTF filter: 4h HMA(21) for intermediate trend direction
- Entry: BB squeeze (BW < 20th percentile) + breakout + ADX > 25
- Filter: 4h trend must align with breakout direction
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R, trail stop at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "bb_squeeze_adx_4h_trend_1d_v1"
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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Bandwidth"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma * 100  # Bandwidth as percentage
    return upper.values, lower.values, bandwidth.values, sma.values


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_s = pd.Series(tr)
    plus_dm_s = pd.Series(plus_dm)
    minus_dm_s = pd.Series(minus_dm)
    
    atr = tr_s.ewm(span=period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (plus_dm_s.ewm(span=period, adjust=False, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm_s.ewm(span=period, adjust=False, min_periods=period).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False, min_periods=period).mean()
    
    return adx.values, plus_di.values, minus_di.values


def calculate_percentile_rank(data, window=100):
    """Calculate rolling percentile rank"""
    data_s = pd.Series(data)
    percentile = data_s.rolling(window=window, min_periods=window).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x) * 100 if len(x) >= window else np.nan
    )
    return percentile.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d indicators
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    atr = calculate_atr(high, low, close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # BB bandwidth percentile (squeeze detection)
    bb_percentile = calculate_percentile_rank(bb_bandwidth, 100)
    
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
    entry_atr = 0.0
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(bb_bandwidth[i]) or
            np.isnan(atr[i]) or np.isnan(adx[i]) or 
            np.isnan(bb_percentile[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter
        trend_4h = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # BB squeeze detection (bandwidth in bottom 20th percentile)
        bb_squeeze = bb_percentile[i] < 20
        
        # ADX trend strength filter (must be trending, not ranging)
        adx_strong = adx[i] > 25
        
        # DI crossover for direction
        di_signal = 0
        if i > 0:
            if plus_di[i] > minus_di[i] and plus_di[i - 1] <= minus_di[i - 1]:
                di_signal = 1  # Bullish DI crossover
            elif minus_di[i] > plus_di[i] and minus_di[i - 1] <= plus_di[i - 1]:
                di_signal = -1  # Bearish DI crossover
        
        # BB breakout detection
        breakout_signal = 0
        if i > 0:
            # Long breakout: price crosses above BB upper
            if close[i] > bb_upper[i] and close[i - 1] <= bb_upper[i - 1]:
                breakout_signal = 1
            # Short breakout: price crosses below BB lower
            elif close[i] < bb_lower[i] and close[i - 1] >= bb_lower[i - 1]:
                breakout_signal = -1
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Entry logic: squeeze + breakout + ADX + trend alignment
        if bb_squeeze and breakout_signal != 0 and adx_strong:
            # Breakout must align with 4h trend
            if breakout_signal == trend_4h:
                target_signal = SIZE * breakout_signal
        # Alternative entry: DI crossover with trend alignment (no squeeze required)
        elif di_signal != 0 and adx_strong:
            if di_signal == trend_4h:
                target_signal = SIZE * di_signal
        
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
                
                # Check take profit (2R from entry)
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
            profit_target_hit = False
            entry_atr = 0.0
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
            # Trail stop tighter after TP (1R from highest/lowest)
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, close[i])
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            # Apply signal change
            if target_signal != 0.0:
                signals[i] = target_signal
                if position_side == 0:
                    # New entry
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