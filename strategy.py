#!/usr/bin/env python3
"""
EXPERIMENT #075 - HMA Trend + RSI Pullback + 4h HTF Filter + Volume Confirmation (1h primary)
==============================================================================================
Hypothesis: 1h HMA trend following with RSI pullback entries captures trend continuations
better than breakouts. 4h HMA filter ensures we trade with the higher timeframe trend.
Volume confirmation (taker buy ratio) reduces false signals during low-liquidity periods.

Key differences from current best (Donchian 12h):
- Entries on pullbacks (RSI 40-60) rather than breakouts (better risk/reward)
- 1h primary captures more opportunities than 12h while maintaining trend quality
- Volume confirmation filters out choppy/low-liquidity periods
- HMA(16/48) crossover is more responsive than Donchian(20) for trend changes

Why this should beat Sharpe=0.490:
- Pullback entries have tighter stops than breakout entries
- 4h HTF filter removes counter-trend trades (major source of losses)
- Volume filter reduces whipsaws during Asian session low-liquidity
- Conservative sizing (0.25-0.30) with 2.5*ATR stop controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_pullback_vol_1h_4h_v1"
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
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    delta = np.zeros(n)
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio"""
    ratio = np.zeros(len(volume))
    for i in range(len(volume)):
        if volume[i] > 0:
            ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            ratio[i] = 0.5
    return ratio


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    taker_buy_volume = prices["taker_buy_volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    hma_fast = calculate_hma(close, 16)
    hma_slow = calculate_hma(close, 48)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
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
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(hma_fast[i]) or 
            np.isnan(hma_slow[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ratio[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HTF trend direction
        htftrend_bullish = close[i] > hma_4h_aligned[i]
        htftrend_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h HMA crossover signals
        hma_crossover_long = hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]
        hma_crossover_short = hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]
        
        # HMA trend confirmation (fast above slow for long, below for short)
        hma_trend_long = hma_fast[i] > hma_slow[i]
        hma_trend_short = hma_fast[i] < hma_slow[i]
        
        # RSI pullback zones (entry on pullback, not extreme)
        rsi_pullback_long = 40 <= rsi[i] <= 60  # Pullback in uptrend
        rsi_pullback_short = 40 <= rsi[i] <= 60  # Pullback in downtrend
        
        # Volume confirmation (taker buy ratio > 0.45 for long, < 0.55 for short)
        vol_confirm_long = vol_ratio[i] > 0.45
        vol_confirm_short = vol_ratio[i] < 0.55
        
        # Calculate position size based on volume strength
        vol_multiplier = 1.0
        if vol_ratio[i] > 0.55:
            vol_multiplier = 1.15  # Strong buying pressure
        elif vol_ratio[i] < 0.45:
            vol_multiplier = 1.15  # Strong selling pressure
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * vol_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: HMA crossover + 4h bullish + RSI pullback + Volume confirmation
        if (hma_crossover_long and htftrend_bullish and 
            rsi_pullback_long and vol_confirm_long):
            target_signal = position_size
        
        # Short entry: HMA crossover + 4h bearish + RSI pullback + Volume confirmation
        elif (hma_crossover_short and htftrend_bearish and 
              rsi_pullback_short and vol_confirm_short):
            target_signal = -position_size
        
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
                # Exit if HMA trend reverses OR 4h HTF alignment breaks
                hma_reversal_long = hma_fast[i] < hma_slow[i]
                hma_reversal_short = hma_fast[i] > hma_slow[i]
                htf_alignment_broken = (position_side == 1 and htftrend_bearish) or \
                                       (position_side == -1 and htftrend_bullish)
                
                if hma_reversal_long or hma_reversal_short or htf_alignment_broken:
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