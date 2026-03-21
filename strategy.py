#!/usr/bin/env python3
"""
EXPERIMENT #009 - Keltner Channel Breakout + Volume Confirmation + Dual HTF Filter (1h primary)
===============================================================================================
Hypothesis: Keltner Channels (ATR-based) provide better breakout signals than Donchian (pure high/low)
because they adapt to volatility. Volume confirmation filters false breakouts. Dual HTF filter
(4h EMA + 1d HMA) ensures we trade with the trend. This differs from failed strategies by:
- Using Keltner (ATR-based) instead of Donchian (fixed lookback)
- Adding volume surge confirmation (not just price breakout)
- Volatility regime filter (BB width percentile) to avoid low-vol traps
- Conservative sizing (0.25-0.30) with proper stoploss

Key features:
- Primary TF: 1h (required for this experiment)
- HTF filters: 4h EMA(50) + 1d HMA(50) for dual alignment
- Trend: Keltner Channel(20, 2.0*ATR) breakout
- Entry: Price breaks Keltner + volume > 1.5x avg + HTF alignment
- Regime: Bollinger Band Width percentile > 40th (avoid ultra-low vol)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete, stoploss at 2*ATR
- Take profit: Reduce to half at 2R profit

Why this should beat current best:
- Keltner adapts to volatility (better than fixed Donchian)
- Volume filter removes 40%+ of false breakouts
- Dual HTF (4h+1d) simpler than triple, less lag
- 1h timeframe balances signal frequency vs noise
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "keltner_volume_dualhtf_1h_4h_1d_v1"
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


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    return ema.values


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


def calculate_keltner(high, low, close, period=20, atr_mult=2.0):
    """Calculate Keltner Channels (EMA middle, ATR-based bands)"""
    n = len(close)
    middle = calculate_ema(close, period)
    atr = calculate_atr(high, low, close, period)
    
    upper = middle + atr_mult * atr
    lower = middle - atr_mult * atr
    
    return upper, lower, middle


def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return upper, lower, middle, std


def calculate_bb_width_percentile(upper, lower, middle, window=100):
    """Calculate Bollinger Band Width and its percentile rank"""
    n = len(upper)
    bb_width = (upper - lower) / middle
    
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(window - 1, n):
        if not np.isnan(bb_width[i]) and not np.isnan(bb_width[i-window+1:i+1]).all():
            window_data = bb_width[i - window + 1:i + 1]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) > 0:
                pr[i] = np.sum(window_data <= bb_width[i]) / len(window_data)
    
    return pr, bb_width


def calculate_volume_sma(volume, period=20):
    """Calculate volume simple moving average"""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    ema_4h = calculate_ema(df_4h['close'].values, 50)
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1h indicators
    keltner_upper, keltner_lower, keltner_middle = calculate_keltner(high, low, close, 20, 2.0)
    bb_upper, bb_lower, bb_middle, bb_std = calculate_bollinger(close, 20, 2.0)
    atr = calculate_atr(high, low, close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    # Calculate BB width percentile (volatility regime filter)
    bb_width_pr, bb_width = calculate_bb_width_percentile(bb_upper, bb_lower, bb_middle, 100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.32   # Max position size
    MIN_SIZE = 0.22   # Min position size
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
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]) or
            np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(atr[i]) or np.isnan(vol_sma[i]) or np.isnan(bb_width_pr[i]) or
            atr[i] == 0 or vol_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Dual HTF trend alignment
        price_above_4h_ema = close[i] > ema_4h_aligned[i]
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # 4h and 1d trend direction
        fourh_trend = 1 if price_above_4h_ema else -1
        daily_trend = 1 if price_above_1d_hma else -1
        
        # Volatility regime filter (avoid ultra-low vol chop)
        vol_regime_ok = bb_width_pr[i] > 0.40  # Above 40th percentile
        
        # Keltner breakout signals
        breakout_long = close[i] > keltner_upper[i - 1]  # Break above previous upper
        breakout_short = close[i] < keltner_lower[i - 1]  # Break below previous lower
        
        # Volume confirmation (volume surge > 1.5x average)
        volume_surge = volume[i] > 1.5 * vol_sma[i]
        
        # Calculate position size (discrete levels to reduce fee churn)
        if vol_regime_ok and volume_surge:
            position_size = MAX_SIZE
        elif vol_regime_ok:
            position_size = BASE_SIZE
        else:
            position_size = MIN_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Keltner breakout + volume surge + Dual HTF bullish
        if (breakout_long and volume_surge and 
            fourh_trend == 1 and daily_trend == 1):
            target_signal = position_size
        
        # Short entry: Keltner breakout + volume surge + Dual HTF bearish
        elif (breakout_short and volume_surge and 
              fourh_trend == -1 and daily_trend == -1):
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
                # Exit if Keltner reverses OR HTF alignment breaks
                keltner_reversal_long = close[i] < keltner_middle[i]
                keltner_reversal_short = close[i] > keltner_middle[i]
                hma_alignment_broken = (position_side == 1 and daily_trend == -1) or \
                                       (position_side == -1 and daily_trend == 1)
                
                if keltner_reversal_long or keltner_reversal_short or hma_alignment_broken:
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