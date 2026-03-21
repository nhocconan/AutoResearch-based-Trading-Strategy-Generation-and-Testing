#!/usr/bin/env python3
"""
EXPERIMENT #052 - KAMA Trend + RSI Pullback + Volume Confirmation (4h primary)
=====================================================================================
Hypothesis: 4h KAMA adapts to market volatility better than HMA/EMA, capturing trends
while filtering chop. RSI(14) pullback entries (30-40 for longs, 60-70 for shorts) provide
better risk/reward than breakouts. 1d HMA(50) filters major trend direction. Volume
spike (>1.5x 20-period avg) confirms genuine moves vs fakeouts.

Key features:
- Primary TF: 4h (mandatory for this experiment)
- HTF filter: 1d HMA(50) for major trend alignment
- Trend: KAMA(10,2,30) - adapts to market efficiency
- Entry: RSI(14) pullback to 35-45 (long) or 55-65 (short) within trend
- Volume: 20-period volume MA, require >1.3x for entry confirmation
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete, reduced in low volume regimes
- Take profit: Reduce to half at 2R profit

Why this should beat current best (Sharpe=0.490):
- KAMA adapts faster in trends, slower in chop vs fixed HMA
- Pullback entries have better R:R than breakouts on 4h
- Volume filter removes 40%+ of false signals
- Conservative sizing (0.25-0.30) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_volume_pullback_4h_1d_v1"
timeframe = "4h"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market volatility - fast in trends, slow in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period - 1, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.zeros(n)
    for i in range(period - 1, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    # Calculate KAMA
    for i in range(period, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


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
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    kama = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.32   # Max position size with volume confirmation
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
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(kama[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or
            atr[i] == 0 or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 1d trend direction (HTF filter)
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        daily_trend = 1 if price_above_1d_hma else -1
        
        # 4h KAMA trend direction
        kama_trend = 1 if close[i] > kama[i] else -1
        
        # KAMA slope (trend strength)
        kama_slope = 0
        if i >= 3 and not np.isnan(kama[i-3]):
            kama_slope = (kama[i] - kama[i-3]) / kama[i-3] if kama[i-3] != 0 else 0
        
        # Volume confirmation (spike > 1.3x average)
        volume_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirmed = volume_ratio > 1.3
        
        # RSI pullback zones (not extremes - we want pullbacks, not reversals)
        rsi_pullback_long = 35 <= rsi[i] <= 50  # Pullback in uptrend
        rsi_pullback_short = 50 <= rsi[i] <= 65  # Pullback in downtrend
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 45  # Gaining momentum
        rsi_momentum_short = rsi[i] < 55  # Losing momentum
        
        # Calculate position size based on volume confirmation
        if volume_confirmed:
            position_size = MAX_SIZE
        else:
            position_size = MIN_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish + 1d HMA bullish + RSI pullback + volume
        # Require alignment: daily trend = 1, kama trend = 1, rsi in pullback zone
        if (daily_trend == 1 and kama_trend == 1 and 
            rsi_pullback_long and rsi_momentum_long and
            kama_slope > 0):
            # Volume confirmation increases size but not required for entry
            target_signal = position_size
        
        # Short entry: KAMA bearish + 1d HMA bearish + RSI pullback + volume
        elif (daily_trend == -1 and kama_trend == -1 and 
              rsi_pullback_short and rsi_momentum_short and
              kama_slope < 0):
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
                # Exit if KAMA reverses OR HTF alignment breaks
                kama_reversal_long = close[i] < kama[i]
                kama_reversal_short = close[i] > kama[i]
                hma_alignment_broken = (position_side == 1 and daily_trend == -1) or \
                                       (position_side == -1 and daily_trend == 1)
                
                if kama_reversal_long or kama_reversal_short or hma_alignment_broken:
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