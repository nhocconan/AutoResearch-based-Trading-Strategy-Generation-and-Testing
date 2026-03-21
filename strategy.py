#!/usr/bin/env python3
"""
EXPERIMENT #018 - HMA Crossover + Z-Score + Weekly Trend Filter (1d primary)
=====================================================================================
Hypothesis: Daily timeframe provides cleaner signals with less noise than intraday.
HMA(8)/HMA(21) crossover captures trend changes quickly. Z-score(20) filter avoids
entering at extreme overbought/oversold levels. Weekly HMA(21) ensures we trade
with the major trend direction. Volume filter confirms breakout validity.

Key features:
- Primary TF: 1d (REQUIRED for this experiment)
- HTF filter: 1w HMA(21) for major trend direction
- Trend: HMA(8)/HMA(21) crossover for entry signals
- Filter: Z-score(20) between -2 and +2 (avoid extremes)
- Volume: Above 20-day average for confirmation
- Stoploss: 2.5*ATR(14) trailing (wider for daily TF)
- Position sizing: 0.25-0.35 discrete levels
- Take profit: Reduce to half at 2R profit

Why this should work on daily:
- Daily bars filter out intraday noise and false breakouts
- Weekly trend filter aligns with major crypto cycles
- Z-score prevents buying tops/selling bottoms
- Conservative sizing controls drawdown during 2022 crash
- Fewer but higher quality trades than intraday strategies
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_zscore_1wtrend_1d_v1"
timeframe = "1d"
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


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma1 = close_s.ewm(span=half_period, adjust=False, min_periods=half_period).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean()
    return hma.values


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized price deviation from mean)"""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std
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
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for major trend filter
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    hma_fast = calculate_hma(close, 8)
    hma_slow = calculate_hma(close, 21)
    atr = calculate_atr(high, low, close, period=14)
    zscore = calculate_zscore(close, period=20)
    volume_ma = calculate_volume_ma(volume, period=20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size with strong confirmation
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 50  # Wait for all indicators to stabilize (daily needs less than intraday)
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(hma_fast[i]) or
            np.isnan(hma_slow[i]) or np.isnan(atr[i]) or np.isnan(zscore[i]) or
            np.isnan(volume_ma[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 1w HMA major trend filter
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        weekly_trend = 1 if price_above_1w_hma else -1
        
        # HMA crossover signals
        hma_bullish_cross = hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]
        hma_bearish_cross = hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]
        hma_bullish = hma_fast[i] > hma_slow[i]
        hma_bearish = hma_fast[i] < hma_slow[i]
        
        # Z-score filter (avoid extremes)
        zscore_ok_long = zscore[i] < 1.5  # Not extremely overbought
        zscore_ok_short = zscore[i] > -1.5  # Not extremely oversold
        zscore_neutral = abs(zscore[i]) < 2.0  # Within normal range
        
        # Volume confirmation
        volume_ok = volume[i] > volume_ma[i] * 0.8  # At least 80% of average volume
        
        # Calculate position size based on ATR (smaller position when volatility is high)
        atr_pct = atr[i] / close[i] * 100  # ATR as percentage of price
        vol_adjustment = 1.0
        if atr_pct > 5.0:  # High volatility day
            vol_adjustment = 0.8
        elif atr_pct < 2.0:  # Low volatility day
            vol_adjustment = 1.1
        
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * vol_adjustment))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: HMA bullish + weekly trend up + Z-score ok + volume ok
        if (hma_bullish and weekly_trend == 1 and zscore_ok_long and volume_ok):
            # Extra confirmation on crossover
            if hma_bullish_cross:
                target_signal = position_size
            elif position_side == 0 and zscore_neutral:
                # Enter on pullback within trend
                target_signal = position_size * 0.8
        
        # Short entry: HMA bearish + weekly trend down + Z-score ok + volume ok
        elif (hma_bearish and weekly_trend == -1 and zscore_ok_short and volume_ok):
            # Extra confirmation on crossover
            if hma_bearish_cross:
                target_signal = -position_size
            elif position_side == 0 and zscore_neutral:
                # Enter on pullback within trend
                target_signal = -position_size * 0.8
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]  # Wider stop for daily
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2R = 5*ATR
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
                    if close[i] <= entry_price - 5.0 * entry_atr:  # 2R profit
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
                # Exit if HMA crosses against position OR weekly trend breaks
                hma_reversal_long = hma_bearish_cross or hma_bearish
                hma_reversal_short = hma_bullish_cross or hma_bullish
                weekly_alignment_broken = (position_side == 1 and weekly_trend == -1) or \
                                          (position_side == -1 and weekly_trend == 1)
                
                # Only exit on strong reversal signals (not just HMA crossing temporarily)
                if weekly_alignment_broken:
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