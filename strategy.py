#!/usr/bin/env python3
"""
EXPERIMENT #018 - EMA Crossover + Weekly Trend + Volume Confirmation (1d primary, 1w HTF)
=========================================================================================
Hypothesis: Daily EMA crossover (12/26) captures medium-term trends, but only when 
aligned with weekly HMA(21) major trend. Volume confirmation (above 20-day avg) 
filters false breakouts. RSI(14) prevents chasing overbought/oversold levels.
This differs from previous attempts by using 1w HTF (not 1d/4h) for stronger trend filter
and volume confirmation which was missing in failed EMA strategies.

Key features:
- Primary TF: 1d (daily candles)
- HTF filter: 1w HMA(21) for major trend direction
- Trend: EMA(12) vs EMA(26) crossover on 1d
- Volume: Current volume > 1.2 * 20-day avg volume
- RSI filter: 30 < RSI < 70 (avoid extremes)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ema_cross_volume_weekly_1d_1w_v1"
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


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    return ema.values


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


def calculate_volume_sma(volume, period=20):
    """Calculate volume simple moving average"""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean()
    return vol_sma.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    ema_fast = calculate_ema(close, 12)
    ema_slow = calculate_ema(close, 26)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
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
    
    min_period = 60  # Wait for all indicators to stabilize (26 EMA + 20 vol + 14 RSI + weekly alignment)
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(ema_fast[i]) or 
            np.isnan(ema_slow[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_sma[i]) or atr[i] == 0 or vol_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter (HTF) - major trend direction
        weekly_trend = 1 if close[i] > hma_1w_aligned[i] else -1
        
        # Daily EMA crossover signal
        ema_cross = 0
        if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
            ema_cross = 1  # Bullish crossover
        elif ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]:
            ema_cross = -1  # Bearish crossover
        
        # Volume confirmation (current volume > 1.2 * 20-day avg)
        volume_confirmed = volume[i] > 1.2 * vol_sma[i]
        
        # RSI filter (avoid overbought/oversold)
        rsi_valid = 30 < rsi[i] < 70
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: EMA bullish crossover + Weekly trend bullish + Volume confirmed + RSI valid
        if ema_cross == 1 and weekly_trend == 1 and volume_confirmed and rsi_valid:
            target_signal = SIZE
        
        # Short entry: EMA bearish crossover + Weekly trend bearish + Volume confirmed + RSI valid
        elif ema_cross == -1 and weekly_trend == -1 and volume_confirmed and rsi_valid:
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
                entry_idx = i
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                if position_side == 1 and ema_fast[i] < ema_slow[i]:
                    # EMA crossed bearish, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and ema_fast[i] > ema_slow[i]:
                    # EMA crossed bullish, exit short
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