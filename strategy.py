#!/usr/bin/env python3
"""
EXPERIMENT #015 - MACD Momentum + Bollinger Squeeze Breakout with 4h Trend Filter (1h)
======================================================================================
Hypothesis: 1h MACD histogram momentum combined with Bollinger Band squeeze detection
captures volatility expansion breakouts when aligned with 4h HMA trend direction.
Volume confirmation filters false breakouts. This differs from previous RSI/KAMA
approaches by using momentum + volatility regime detection instead of mean reversion.

Key features:
- Primary TF: 1h (hourly candles)
- HTF filter: 4h HMA(21) for trend direction
- Entry: MACD histogram expansion + Bollinger squeeze breakout
- Filter: 4h trend must align with breakout direction
- Volume: must be > 20-period average
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this should work:
- MACD histogram captures momentum shifts before price
- Bollinger squeeze identifies low-volatility compression before expansion
- 4h trend filter prevents counter-trend trades
- Volume confirmation reduces false breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "macd_bb_squeeze_4h_trend_1h_v1"
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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_s.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values


def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma  # Normalized bandwidth for squeeze detection
    return upper.values, lower.values, bandwidth.values


def calculate_volume_sma(volume, period=20):
    """Calculate volume simple moving average"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values


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
    
    # Calculate 1h indicators
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger(close, 20, 2.0)
    atr = calculate_atr(high, low, close, 14)
    volume_sma = calculate_volume_sma(volume, 20)
    
    # Calculate Bollinger Band squeeze threshold (bottom 20% of historical bandwidth)
    bb_bandwidth_s = pd.Series(bb_bandwidth)
    bb_squeeze_threshold = bb_bandwidth_s.rolling(window=100, min_periods=50).quantile(0.20).values
    
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
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(macd_hist[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_bandwidth[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_sma[i]) or np.isnan(bb_squeeze_threshold[i]) or 
            atr[i] == 0 or bb_squeeze_threshold[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter
        trend_4h = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # Volume confirmation (must be above 20-period average)
        volume_confirmed = volume[i] > volume_sma[i]
        
        # Bollinger squeeze detection (bandwidth below 20th percentile)
        squeeze_active = bb_bandwidth[i] < bb_squeeze_threshold[i]
        
        # Bollinger breakout detection
        breakout_signal = 0
        if i > 0:
            # Long breakout: price crosses above upper band
            if close[i] > bb_upper[i] and close[i - 1] <= bb_upper[i - 1]:
                breakout_signal = 1
            # Short breakout: price crosses below lower band
            elif close[i] < bb_lower[i] and close[i - 1] >= bb_lower[i - 1]:
                breakout_signal = -1
        
        # MACD histogram momentum confirmation
        macd_momentum = 0
        if i > 1:
            # Long momentum: histogram increasing and positive
            if macd_hist[i] > 0 and macd_hist[i] > macd_hist[i - 1]:
                macd_momentum = 1
            # Short momentum: histogram decreasing and negative
            elif macd_hist[i] < 0 and macd_hist[i] < macd_hist[i - 1]:
                macd_momentum = -1
        
        # Determine target signal based on all filters
        target_signal = 0.0
        if breakout_signal != 0:
            # Breakout must align with 4h trend
            if breakout_signal == trend_4h:
                # Require volume confirmation
                if volume_confirmed:
                    # Require MACD momentum alignment
                    if macd_momentum == breakout_signal:
                        # Bonus: squeeze breakout gets full size
                        if squeeze_active:
                            target_signal = SIZE * breakout_signal
                        else:
                            # Non-squeeze breakout gets half size
                            target_signal = (SIZE / 2) * breakout_signal
        
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
                
                # Check take profit (2R from entry)
                if not profit_target_hit:
                    risk = entry_price - (entry_price - entry_atr * 2.0)
                    if close[i] >= entry_price + 2.0 * entry_atr * 2.0:  # 2R = 4*ATR
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
                    if close[i] <= entry_price - 2.0 * entry_atr * 2.0:  # 2R profit
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
                # Maintain existing position
                if profit_target_hit:
                    signals[i] = HALF_SIZE * position_side
                else:
                    signals[i] = SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals