#!/usr/bin/env python3
"""
EXPERIMENT #027 - HMA Trend + RSI/Z-Score Mean Reversion + Volume (1h primary, 12h HTF)
========================================================================================
Hypothesis: 12h HMA(50) provides stable major trend direction. On 1h timeframe,
we enter on RSI extremes (30/70) ONLY when Z-score confirms price is at statistical
extreme (>2 std dev from 20-bar mean). Volume spike (>1.5x 20-bar avg) confirms
institutional participation. This differs from previous attempts by combining
RSI extremes with Z-score confirmation - both must agree for entry.

Key features:
- Primary TF: 1h (this experiment)
- HTF filter: 12h HMA(50) for major trend direction
- Entry: RSI(14) < 30 (long) or > 70 (short) + Z-score(20) > 2 or < -2
- Volume confirmation: volume > 1.5x 20-bar average
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25 base, 0.125 half (discrete levels)
- Take profit: Reduce to half at 2R profit, trail stop at 1R

Why this should work:
- 12h HMA is slower/more stable than 4h (fewer whipsaws)
- RSI + Z-score double confirmation reduces false signals
- Volume filter ensures we trade with institutional flow
- Conservative sizing (0.25) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_zscore_volume_1h_12h_v1"
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


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized distance from mean)"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / (std + 1e-10)
    return zscore.values


def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average"""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / (vol_avg + 1e-10)
    return vol_ratio.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1)
    df_12h = get_htf_data(prices, '12h')
    hma_12h = calculate_hma(df_12h['close'].values, 50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size (25% of capital)
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
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(rsi[i]) or np.isnan(zscore[i]) or np.isnan(vol_ratio[i]) or 
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 12h trend filter (HTF)
        daily_trend = 1 if close[i] > hma_12h_aligned[i] else -1
        
        # Volume confirmation (must be > 1.5x average)
        volume_confirmed = vol_ratio[i] > 1.5
        
        # RSI extremes
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Z-score extremes (price at statistical extreme)
        zscore_low = zscore[i] < -2.0
        zscore_high = zscore[i] > 2.0
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: 12h trend bullish + RSI oversold + Z-score low + Volume confirmed
        if daily_trend == 1 and rsi_oversold and zscore_low and volume_confirmed:
            target_signal = SIZE
        
        # Short entry: 12h trend bearish + RSI overbought + Z-score high + Volume confirmed
        elif daily_trend == -1 and rsi_overbought and zscore_high and volume_confirmed:
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
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                if position_side == 1 and daily_trend == -1:
                    # Trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and daily_trend == 1:
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