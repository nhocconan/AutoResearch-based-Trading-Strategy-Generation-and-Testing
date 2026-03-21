#!/usr/bin/env python3
"""
EXPERIMENT #003 - HMA Trend + RSI Pullback with 4h Filter (1h)
=================================================================
Hypothesis: 1h timeframe with 4h HMA trend filter provides optimal balance
between trade frequency and signal quality. RSI pullback entries in direction
of HTF trend capture swing moves while ATR trailing stop limits drawdown.

Key features:
- Primary TF: 1h (mandatory for this experiment)
- HTF filter: 4h HMA(21) for trend direction
- Entry: RSI(14) pullback to 35-55 zone in trend direction
- Filter: Price above/below 4h HMA confirms trend
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels to minimize fee churn

Why 1h: More trades than 4h, less noise than 15m/30m. Should generate 10+ trades.
Learning from #002 crash: Simplified indicator calculations, ensure no data issues.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_4h_trend_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average for trend direction"""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_rsi(close, period=14):
    """Calculate RSI oscillator"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_sma(close, period):
    """Calculate Simple Moving Average"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)  # auto shift(1)
    
    # Calculate 1h indicators (all before loop for performance)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30  # Entry position size (30% of capital)
    SIZE_HALF = 0.15   # Half position for take profit
    SIZE_EXIT = 0.0    # Flat
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    take_profit_hit = False
    entry_atr = 0.0
    
    min_period = 220  # Wait for SMA200 and all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN or zero in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or np.isnan(sma_50[i]) or np.isnan(sma_200[i]) or 
            atr[i] <= 0 or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h Trend filter (HTF alignment ensures no look-ahead)
        trend_4h = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # 1h trend filter (price above/below SMA50 and SMA200)
        trend_1h = 1 if (close[i] > sma_50[i] and close[i] > sma_200[i]) else -1
        if close[i] < sma_50[i] and close[i] < sma_200[i]:
            trend_1h = -1
        
        # RSI pullback zone (allows entry on pullback, not at extremes)
        # Long: RSI between 35-55 (pullback in uptrend)
        # Short: RSI between 45-65 (pullback in downtrend)
        rsi_long_ok = 35 <= rsi[i] <= 55
        rsi_short_ok = 45 <= rsi[i] <= 65
        
        # Determine target signal based on ensemble
        target_signal = 0.0
        
        # Long entry: 4h trend up + 1h trend up + RSI in pullback zone
        if trend_4h == 1 and trend_1h == 1 and rsi_long_ok:
            target_signal = SIZE_ENTRY
        
        # Short entry: 4h trend down + 1h trend down + RSI in pullback zone
        elif trend_4h == -1 and trend_1h == -1 and rsi_short_ok:
            target_signal = -SIZE_ENTRY
        
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
                
                # Check take profit (2R profit = entry + 2*ATR at entry)
                if entry_atr > 0:
                    profit_target = entry_price + 2.0 * entry_atr
                    if not take_profit_hit and close[i] >= profit_target:
                        take_profit_triggered = True
                        take_profit_hit = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R profit = entry - 2*ATR at entry)
                if entry_atr > 0:
                    profit_target = entry_price - 2.0 * entry_atr
                    if not take_profit_hit and close[i] <= profit_target:
                        take_profit_triggered = True
                        take_profit_hit = True
        
        if stoploss_triggered:
            signals[i] = SIZE_EXIT
            position_side = 0
            entry_price = 0.0
            entry_atr = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            take_profit_hit = False
        elif take_profit_triggered:
            # Reduce to half position at 2R profit
            signals[i] = SIZE_HALF * position_side
            # Position remains open but reduced, trail stop continues
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                take_profit_hit = False
            elif position_side != 0:
                # Maintain existing position
                if target_signal == 0.0:
                    # Exit signal (trend changed)
                    signals[i] = SIZE_EXIT
                    position_side = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    take_profit_hit = False
                elif np.sign(target_signal) == position_side:
                    # Same direction - maintain position
                    signals[i] = SIZE_ENTRY * position_side
                # else: ignore opposite signal (no flipping without exit)
            else:
                signals[i] = SIZE_EXIT
    
    return signals