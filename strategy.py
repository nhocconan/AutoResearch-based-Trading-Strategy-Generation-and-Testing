#!/usr/bin/env python3
"""
EXPERIMENT #003 - KAMA Adaptive Trend + MACD Momentum + Volume Filter (1h primary)
=====================================================================================
Hypothesis: KAMA adapts to market volatility better than fixed EMAs, reducing whipsaws
in chop while capturing trends efficiently. MACD histogram confirms momentum direction.
Volume filter ensures breakouts have participation. 4h HMA provides trend alignment.

Key features:
- Primary TF: 1h
- HTF filter: 4h HMA(21) for trend direction
- Trend: KAMA(14) adaptive moving average
- Momentum: MACD(12,26,9) histogram
- Volume: Volume > 1.5x 20-period average for confirmation
- Entry: KAMA crossover + MACD confirmation + Volume spike + HTF alignment
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R, trail stop

Why this should beat current best:
- KAMA adapts to volatility (better than fixed HMA/EMA in chop)
- Volume filter reduces false breakouts
- 4h trend filter ensures we trade with major trend
- Conservative sizing (0.28 base) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_macd_volume_4hfilter_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period + 10:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
            continue
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD and histogram"""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_s.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values


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
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    kama = calculate_kama(close, 14)
    macd_line, signal_line, macd_hist = calculate_macd(close, 12, 26, 9)
    atr = calculate_atr(high, low, close, 14)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 100
    
    for i in range(min_period, n):
        # Check for NaN
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(kama[i]) or
            np.isnan(macd_hist[i]) or np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or atr[i] == 0 or kama[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter
        ht_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # KAMA trend
        kama_trend = 1 if close[i] > kama[i] else -1
        kama_slope = kama[i] - kama[i - 1] if i > 0 else 0
        
        # MACD momentum (histogram increasing/decreasing)
        macd_bullish = macd_hist[i] > 0 and (i < 1 or macd_hist[i] > macd_hist[i - 1])
        macd_bearish = macd_hist[i] < 0 and (i < 1 or macd_hist[i] < macd_hist[i - 1])
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Calculate target signal
        target_signal = 0.0
        
        # Long entry: 4h bullish + KAMA bullish + MACD bullish + Volume
        if (ht_trend == 1 and kama_trend == 1 and kama_slope > 0 and
            macd_bullish and volume_confirmed):
            target_signal = BASE_SIZE
        
        # Short entry: 4h bearish + KAMA bearish + MACD bearish + Volume
        elif (ht_trend == -1 and kama_trend == -1 and kama_slope < 0 and
              macd_bearish and volume_confirmed):
            target_signal = -BASE_SIZE
        
        # Stoploss and take profit logic
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * entry_atr:
                        take_profit_triggered = True
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * entry_atr:
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
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            if target_signal != 0.0 and position_side == 0:
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                kama_reversal_long = close[i] < kama[i] and kama_slope < 0
                kama_reversal_short = close[i] > kama[i] and kama_slope > 0
                hma_alignment_broken = (position_side == 1 and ht_trend == -1) or \
                                       (position_side == -1 and ht_trend == 1)
                
                if kama_reversal_long or kama_reversal_short or hma_alignment_broken:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    signals[i] = BASE_SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals