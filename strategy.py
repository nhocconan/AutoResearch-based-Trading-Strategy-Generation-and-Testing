#!/usr/bin/env python3
"""
EXPERIMENT #050 - HMA Trend + RSI Pullback + BB Regime Filter (30m primary)
=====================================================================================
Hypothesis: 30m timeframe captures medium-term swings better than 15m (less noise) 
and 1h (faster entries). Using 4h HMA(50) for trend direction + RSI(14) pullback 
entries + Bollinger Band width regime filter should outperform pure breakout strategies.

Key features:
- Primary TF: 30m (this experiment's requirement)
- HTF filter: 4h HMA(50) for trend direction, 1d HMA(50) for major trend confirmation
- Entry: RSI(14) pullback in direction of 4h trend (RSI<45 for long, RSI>55 for short)
- Regime: BB width percentile > 40th (avoid low-volatility chop)
- Stoploss: 2.0*ATR(14) trailing stop
- Position sizing: 0.25 base, 0.30 max on strong signals, discrete levels
- Take profit: Reduce to half at 2R profit, trail stop at 1R

Why this should beat current best (Sharpe=0.490):
- 30m captures swings faster than 12h while filtering 15m noise
- RSI pullback entries have better risk/reward than breakouts (enter on dips)
- BB regime filter avoids 40%+ of choppy periods that kill trend strategies
- Conservative sizing (0.25-0.30) with mandatory stoploss controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_bb_regime_30m_4h_1d_v1"
timeframe = "30m"
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
    delta = np.zeros(n)
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rsi = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma


def calculate_bb_width(upper, lower, sma):
    """Calculate Bollinger Band Width (normalized)"""
    width = np.zeros(len(upper))
    for i in range(len(upper)):
        if sma[i] > 0:
            width[i] = (upper[i] - lower[i]) / sma[i]
        else:
            width[i] = 0
    return width


def calculate_percentile_rank(series, window=100):
    """Calculate rolling percentile rank"""
    n = len(series)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(window - 1, n):
        if not np.isnan(series[i]) and series[i] != 0:
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
    hma_4h = calculate_hma(df_4h['close'].values, 50)
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_sma)
    bb_width_pr = calculate_percentile_rank(bb_width, 100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size with strong signals
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
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(bb_width_pr[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # HTF trend alignment
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # 4h and 1d trend direction
        fourh_trend = 1 if price_above_4h_hma else -1
        oned_trend = 1 if price_above_1d_hma else -1
        
        # BB regime filter (avoid low volatility chop)
        bb_regime_ok = bb_width_pr[i] > 0.40  # Only trade when BB width > 40th percentile
        
        # RSI pullback signals
        rsi_oversold = rsi[i] < 45  # Pullback in uptrend
        rsi_overbought = rsi[i] > 55  # Pullback in downtrend
        
        # Calculate position size based on trend alignment strength
        position_size = BASE_SIZE
        if fourh_trend == oned_trend:  # Both HTF agree
            position_size = MAX_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: 4h uptrend + RSI pullback + BB regime ok
        if (fourh_trend == 1 and rsi_oversold and bb_regime_ok):
            target_signal = position_size
        
        # Short entry: 4h downtrend + RSI pullback + BB regime ok
        elif (fourh_trend == -1 and rsi_overbought and bb_regime_ok):
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
                # Exit if 4h HMA trend reverses
                hma_trend_reversed = (position_side == 1 and fourh_trend == -1) or \
                                     (position_side == -1 and fourh_trend == 1)
                
                if hma_trend_reversed:
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