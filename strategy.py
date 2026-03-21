#!/usr/bin/env python3
"""
EXPERIMENT #042 - HMA Trend + RSI Pullback + Weekly Filter (1d primary, 1w HTF)
================================================================================
Hypothesis: Daily HMA(21) captures intermediate trend, RSI(14) pullback to 40-60 
zone provides optimal entry timing, and Weekly HMA(50) ensures we trade with the 
major trend. This simpler approach avoids overfitting from complex multi-HTF setups 
that failed in experiments #030-#041. Daily timeframe reduces noise vs lower TFs.

Key features:
- Primary TF: 1d (daily candles - REQUIRED for this experiment)
- HTF filter: 1w HMA(50) for major trend alignment
- Trend: HMA(21) on 1d for direction
- Entry: RSI(14) pullback to 40-60 zone in trend direction
- Volume filter: volume > 20-day SMA (confirms interest)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels (CRITICAL for DD control)
- Take profit: Reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_weekly_filter_1d_1w_v1"
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


def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback periods"""
    n = len(hma_values)
    slope = np.zeros(n)
    slope[:] = np.nan
    for i in range(lookback - 1, n):
        if not np.isnan(hma_values[i]) and not np.isnan(hma_values[i - lookback + 1]):
            slope[i] = (hma_values[i] - hma_values[i - lookback + 1]) / hma_values[i - lookback + 1]
    return slope


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    hma_1w = calculate_hma(df_1w['close'].values, 50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    hma_slope = calculate_hma_slope(hma_21, 5)
    
    # Volume SMA for confirmation
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital - conservative)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(hma_21[i]) or 
            np.isnan(hma_50[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]) or 
            np.isnan(hma_slope[i]) or np.isnan(volume_sma[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter (HTF) - must align with major trend
        weekly_trend = 1 if close[i] > hma_1w_aligned[i] else -1
        
        # Daily trend filter
        daily_trend = 1 if hma_21[i] > hma_50[i] else -1
        
        # HMA slope confirmation (trend strength)
        slope_positive = hma_slope[i] > 0.001  # 0.1% per 5 days
        slope_negative = hma_slope[i] < -0.001
        
        # Volume confirmation
        volume_confirmed = volume[i] > volume_sma[i]
        
        # RSI pullback zone (40-60 for entry timing)
        rsi_pullback_long = 40 <= rsi[i] <= 60
        rsi_pullback_short = 40 <= rsi[i] <= 60
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Weekly trend bullish + Daily trend bullish + RSI pullback + Volume confirmed
        if (weekly_trend == 1 and daily_trend == 1 and slope_positive and 
            rsi_pullback_long and volume_confirmed):
            target_signal = SIZE
        
        # Short entry: Weekly trend bearish + Daily trend bearish + RSI pullback + Volume confirmed
        elif (weekly_trend == -1 and daily_trend == -1 and slope_negative and 
              rsi_pullback_short and volume_confirmed):
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
                
                # Check take profit (2R from entry, where R = 2.5*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * atr[entry_idx if 'entry_idx' in dir() else i]:
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
                    if close[i] <= entry_price - 5.0 * atr[entry_idx if 'entry_idx' in dir() else i]:
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
                if position_side == 1 and (daily_trend == -1 or weekly_trend == -1):
                    # Trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and (daily_trend == 1 or weekly_trend == 1):
                    # Trend reversed, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals