#!/usr/bin/env python3
"""
EXPERIMENT #003 - KAMA Adaptive Trend + Z-Score Pullback + 12h HTF Filter (1h primary)
======================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better 
than EMA/HMA, reducing whipsaws in choppy markets. Combined with Z-score pullback 
entries in the direction of the 12h major trend, this should capture trends while 
avoiding false breakouts. 12h HMA(50) provides stable major trend filter (less noisy 
than 4h). Volume confirmation ensures we only trade meaningful moves.

Key features:
- Primary TF: 1h (required for this experiment)
- HTF filter: 12h HMA(50) for major trend direction (more stable than 4h)
- Trend: KAMA(21) adaptive moving average on 1h
- Entry: Z-score(20) pullback to -1.0 to +1.0 zone in trend direction
- Confirmation: Volume > 20-period MA (confirms genuine moves)
- Stoploss: 2.5*ATR(14) trailing stop
- Position sizing: 0.25-0.30 discrete levels (conservative to control DD)
- Take profit: Reduce to half at 2R profit, trail stop at 1R

Why different from failed attempts:
- Failed supertrend_rsi had DD=-47% (too aggressive entries)
- Failed hma_kama had DD=-70% (no proper HTF filter, too many signal changes)
- This uses 12h HTF (more stable), Z-score for timing (not RSI), volume filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_zscore_volume_1h_12h_v1"
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


def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market volatility - moves fast in trends, slow in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized price deviation from mean)"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / (std + 1e-10)
    return zscore.values


def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average"""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    hma_12h = calculate_hma(df_12h['close'].values, 50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    kama = calculate_kama(close, 21, 2, 30)
    atr = calculate_atr(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    vol_ma = calculate_volume_ma(volume, 20)
    
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
    entry_atr = 0.0
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(atr[i]) or np.isnan(zscore[i]) or np.isnan(vol_ma[i]) or 
            atr[i] == 0 or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 12h major trend filter (HTF)
        if close[i] > hma_12h_aligned[i]:
            major_trend = 1  # Bullish
        else:
            major_trend = -1  # Bearish
        
        # 1h KAMA trend direction
        if kama[i] > kama[i - 1]:
            kama_trend = 1  # Rising
        else:
            kama_trend = -1  # Falling
        
        # Volume confirmation (volume > 20-period MA)
        volume_confirmed = volume[i] > vol_ma[i]
        
        # Z-score pullback zone (enter on pullbacks in trend direction)
        # For longs: Z-score between -1.5 and 0.5 (pullback but not oversold)
        # For shorts: Z-score between -0.5 and 1.5 (pullback but not overbought)
        zscore_pullback_long = -1.5 <= zscore[i] <= 0.5
        zscore_pullback_short = -0.5 <= zscore[i] <= 1.5
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Major trend bullish + KAMA rising + Z-score pullback + Volume confirmed
        if major_trend == 1 and kama_trend == 1 and zscore_pullback_long and volume_confirmed:
            target_signal = SIZE
        
        # Short entry: Major trend bearish + KAMA falling + Z-score pullback + Volume confirmed
        elif major_trend == -1 and kama_trend == -1 and zscore_pullback_short and volume_confirmed:
            target_signal = -SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        trend_reversal = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR)
                if not profit_target_hit and entry_atr > 0:
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2R = 5*ATR
                        take_profit_triggered = True
                
                # Check trend reversal (KAMA flipped)
                if kama_trend == -1:
                    trend_reversal = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit and entry_atr > 0:
                    if close[i] <= entry_price - 5.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
                
                # Check trend reversal (KAMA flipped)
                if kama_trend == 1:
                    trend_reversal = True
        
        # Apply signal based on conditions
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
        elif trend_reversal:
            # Trend reversed, exit position
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
            entry_atr = 0.0
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
                entry_atr = atr[i]
            elif position_side != 0:
                # Maintain existing position
                if profit_target_hit:
                    signals[i] = HALF_SIZE * position_side
                else:
                    signals[i] = SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals