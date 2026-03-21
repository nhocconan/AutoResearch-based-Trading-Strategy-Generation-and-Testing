#!/usr/bin/env python3
"""
EXPERIMENT #003 - EMA Crossover + Bollinger Regime + 4h Trend Filter (1h)
=========================================================================
Hypothesis: Simple EMA crossovers with Bollinger Band regime detection
provide more reliable signals than adaptive averages. On 1h timeframe,
we enter when fast EMA crosses slow EMA, confirmed by Bollinger Band
expansion (trend regime) and aligned with 4h trend direction.

Key features:
- Primary TF: 1h (required for this experiment)
- HTF filter: 4h EMA(21) for trend direction
- Entry: EMA(8) crosses EMA(21) with BB regime confirmation
- Regime filter: Bollinger Width > median = trending market
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.30 discrete levels

Why different from failed strategies:
- Simpler EMA vs complex KAMA (less calculation errors)
- 1h TF = more stable than 30m/15m, more trades than 4h+
- BB regime filter avoids mean-reversion chop
- Proven EMA crossover baseline with added filters
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ema_bb_regime_4hfilter_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    return ema.values


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands with proper min_periods"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std * std_mult)
    lower = sma - (std * std_mult)
    width = (upper - lower) / sma  # Bollinger Width (normalized)
    return upper.values, lower.values, width.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR with proper min_periods"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    ema_4h = calculate_ema(df_4h['close'].values, 21)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)  # auto shift(1)
    
    # Calculate 1h indicators
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Calculate Bollinger Width median for regime filter
    bb_width_median = np.nanmedian(bb_width[50:])  # Skip initial NaN period
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # Base position size (30% of capital)
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    
    min_period = 50  # Wait for indicators to stabilize
    
    # Track EMA crossover state
    prev_ema_diff = 0.0
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_fast[i]) or 
            np.isnan(ema_slow[i]) or np.isnan(bb_width[i]) or 
            np.isnan(atr[i]) or np.isnan(rsi[i]) or atr[i] == 0):
            signals[i] = 0.0
            prev_ema_diff = ema_fast[i] - ema_slow[i] if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]) else prev_ema_diff
            continue
        
        # 4h Trend filter (HTF)
        htf_trend = 0
        if close[i] > ema_4h_aligned[i]:
            htf_trend = 1  # Bullish HTF
        elif close[i] < ema_4h_aligned[i]:
            htf_trend = -1  # Bearish HTF
        
        # 1h EMA crossover signal
        ema_diff = ema_fast[i] - ema_slow[i]
        ema_cross_long = (prev_ema_diff <= 0) and (ema_diff > 0)  # Fast crosses above Slow
        ema_cross_short = (prev_ema_diff >= 0) and (ema_diff < 0)  # Fast crosses below Slow
        prev_ema_diff = ema_diff
        
        # Bollinger Band regime filter (trending vs mean-reversion)
        # Width > median = trending market (good for breakout strategies)
        # Width < median = mean-reversion market (avoid trend trades)
        regime_trending = bb_width[i] > bb_width_median
        
        # RSI filter to avoid extreme entries
        rsi_valid_long = rsi[i] < 70  # Not overbought
        rsi_valid_short = rsi[i] > 30  # Not oversold
        
        # Determine target signal
        target_signal = 0.0
        
        # Long entry: HTF bullish + EMA cross long + trending regime + RSI valid
        if htf_trend == 1 and ema_cross_long and regime_trending and rsi_valid_long:
            target_signal = BASE_SIZE
        
        # Short entry: HTF bearish + EMA cross short + trending regime + RSI valid
        elif htf_trend == -1 and ema_cross_short and regime_trending and rsi_valid_short:
            target_signal = -BASE_SIZE
        
        # Stoploss logic - check BEFORE setting new signal
        stoploss_triggered = False
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                if close[i] < trailing_stop:
                    stoploss_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                if close[i] > trailing_stop:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
        else:
            # Apply signal change
            if target_signal != 0.0:
                # Check if this is a reversal or new entry
                if position_side == 0:
                    # New entry
                    signals[i] = target_signal
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
                elif np.sign(target_signal) == position_side:
                    # Same direction - maintain position
                    signals[i] = target_signal
                    # Update extremes for trailing stop
                    if position_side == 1:
                        highest_since_entry = max(highest_since_entry, close[i])
                    else:
                        lowest_since_entry = min(lowest_since_entry, close[i])
                else:
                    # Reversal - close old, open new
                    signals[i] = target_signal
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
            elif position_side != 0:
                # Maintain existing position (no new signal but still in trade)
                signals[i] = BASE_SIZE * position_side
                # Update extremes for trailing stop
                if position_side == 1:
                    highest_since_entry = max(highest_since_entry, close[i])
                else:
                    lowest_since_entry = min(lowest_since_entry, close[i])
            else:
                signals[i] = 0.0
    
    return signals