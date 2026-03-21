#!/usr/bin/env python3
"""
EXPERIMENT #025 - EMA Crossover + Z-Score Filter + 1h HMA Trend (15m primary)
=====================================================================================
Hypothesis: 15m EMA(8/21) crossover captures momentum moves, but generates false signals
at extremes. Adding Z-score(20) filter prevents entering when price is >2 std dev from mean.
1h HMA(21) trend filter ensures we trade with higher timeframe direction (less restrictive
than 4h, allowing more trades). Volume spike confirmation adds conviction to entries.

Key features:
- Primary TF: 15m
- HTF filter: 1h HMA(21) for trend direction (more trades than 4h filter)
- Trend: EMA(8) vs EMA(21) crossover
- Filter: Z-score(20) between -1.5 and +1.5 (avoid extremes)
- Confirmation: Volume > 1.5 * SMA(volume, 20)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this should beat failed strategies:
- 1h filter is less restrictive than 4h (failed #021, #024 had too few trades)
- Z-score prevents buying tops/selling bottoms (reduces drawdown)
- Volume confirmation adds signal quality
- Conservative sizing controls drawdown during crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ema_zscore_1hhma_15m_v1"
timeframe = "15m"
leverage = 1.0


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
    ema = close_s.ewm(span=period, adjust=False, min_periods=period).mean().values
    return ema


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized price deviation from mean)"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / std
    return zscore.values


def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume"""
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
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 1h HMA for trend filter
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    
    # Calculate 15m indicators
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    atr = calculate_atr(high, low, close, period=14)
    zscore = calculate_zscore(close, period=20)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size with strong volume
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1h_aligned[i]) or np.isnan(ema_fast[i]) or
            np.isnan(ema_slow[i]) or np.isnan(atr[i]) or np.isnan(zscore[i]) or
            np.isnan(vol_sma[i]) or atr[i] == 0 or vol_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 1h HMA trend filter
        price_above_1h_hma = close[i] > hma_1h_aligned[i]
        hma_trend = 1 if price_above_1h_hma else -1
        
        # EMA crossover signal
        ema_bullish = ema_fast[i] > ema_slow[i]
        ema_bearish = ema_fast[i] < ema_slow[i]
        
        # EMA crossover detection (fast crosses above/below slow)
        ema_cross_long = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_short = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # Z-score filter (avoid extremes)
        zscore_normal = -1.5 < zscore[i] < 1.5
        
        # Volume confirmation (volume > 1.5 * average)
        volume_spike = volume[i] > 1.5 * vol_sma[i]
        
        # Calculate position size based on volume strength
        vol_multiplier = 1.0
        if volume[i] > 2.0 * vol_sma[i]:
            vol_multiplier = 1.2
        elif volume[i] > 1.5 * vol_sma[i]:
            vol_multiplier = 1.1
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * vol_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: EMA bullish + 1h HMA bullish + Z-score normal + (crossover OR volume spike)
        if (ema_bullish and hma_trend == 1 and zscore_normal):
            if ema_cross_long or volume_spike:
                target_signal = position_size
        
        # Short entry: EMA bearish + 1h HMA bearish + Z-score normal + (crossover OR volume spike)
        elif (ema_bearish and hma_trend == -1 and zscore_normal):
            if ema_cross_short or volume_spike:
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
                # Exit if EMA reverses OR 1h HMA alignment breaks
                ema_reversal_long = ema_bearish
                ema_reversal_short = ema_bullish
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
                if ema_reversal_long or ema_reversal_short or hma_alignment_broken:
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