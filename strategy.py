#!/usr/bin/env python3
"""
EXPERIMENT #005 - KAMA Adaptive Trend + RSI Pullback + Daily Trend Filter (12h)
===============================================================================
Hypothesis: 12h timeframe captures medium-term trends better than 1h/4h while
avoiding daily noise. KAMA(10) adapts to volatility (fast in trends, slow in chop).
RSI(14) pullback entries (buy dips in uptrend, sell rallies in downtrend) provide
better risk/reward than breakouts. 1d HMA(50) filters major trend direction.
Bollinger Band width percentile avoids trading in ranging markets.

Key features:
- Primary TF: 12h (this experiment's requirement)
- HTF filter: 1d HMA(50) for major trend direction
- Entry: KAMA(10) crossover + RSI(14) pullback (30-70 range)
- Regime filter: Bollinger BW > 40th percentile (trending market)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_pullback_daily_filter_12h_v1"
timeframe = "12h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market noise: fast in trends, slow in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] ** 2 * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bw = (upper - lower) / sma  # Band Width as % of price
    return upper.values, lower.values, bw.values


def calculate_bw_percentile(bw, lookback=100):
    """Calculate Bollinger Band Width percentile over lookback period"""
    n = len(bw)
    bw_pct = np.zeros(n)
    bw_pct[:] = np.nan
    
    for i in range(lookback, n):
        window = bw[i - lookback:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            bw_pct[i] = np.sum(valid <= bw[i]) / len(valid) * 100
        else:
            bw_pct[i] = 50.0
    
    return bw_pct


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_bw = calculate_bollinger_bands(close, 20, 2.0)
    bb_bw_pct = calculate_bw_percentile(bb_bw, 100)
    
    # KAMA signal (price vs KAMA)
    kama_signal = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(kama[i]) and not np.isnan(kama[i - 1]):
            if close[i] > kama[i] and close[i - 1] <= kama[i - 1]:
                kama_signal[i] = 1  # Bullish crossover
            elif close[i] < kama[i] and close[i - 1] >= kama[i - 1]:
                kama_signal[i] = -1  # Bearish crossover
    
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
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(bb_bw_pct[i]) or
            atr[i] == 0 or bb_bw_pct[i] < 0):
            signals[i] = 0.0
            continue
        
        # Daily trend filter (HTF)
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        
        # Bollinger Band Width regime filter (avoid choppy markets)
        # Only trade when BW percentile > 40 (trending regime)
        trending_regime = bb_bw_pct[i] > 40.0
        
        # RSI pullback filter
        # Long: RSI between 35-60 (pullback in uptrend, not overbought)
        # Short: RSI between 40-65 (rally in downtrend, not oversold)
        rsi_long_valid = 35 < rsi[i] < 60
        rsi_short_valid = 40 < rsi[i] < 65
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish + daily uptrend + trending regime + RSI pullback
        if kama_signal[i] == 1 and daily_trend == 1 and trending_regime and rsi_long_valid:
            target_signal = SIZE
        
        # Short entry: KAMA bearish + daily downtrend + trending regime + RSI rally
        elif kama_signal[i] == -1 and daily_trend == -1 and trending_regime and rsi_short_valid:
            target_signal = -SIZE
        
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
                
                # Check take profit (2R from entry, R = 2.5*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * atr[i]:  # 2R = 5*ATR
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
                    if close[i] <= entry_price - 5.0 * atr[i]:  # 2R profit
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
            if target_signal != 0.0:
                signals[i] = target_signal
                if position_side == 0:
                    # New entry
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
                    profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals