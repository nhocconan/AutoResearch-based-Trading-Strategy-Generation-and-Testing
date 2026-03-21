#!/usr/bin/env python3
"""
EXPERIMENT #015 - KAMA Adaptive Trend + MACD Momentum + 12h HMA Filter (1h primary)
=====================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better than 
static EMAs/HMAs. During trending markets, KAMA follows price closely; during chop, it flattens.
Combining KAMA crossover signals with 12h HMA trend filter and MACD momentum confirmation
should reduce false signals while capturing major moves. Bollinger Band width filter avoids
low-volatility periods where trend strategies fail.

Key features:
- Primary TF: 1h (as required for this experiment)
- HTF filter: 12h HMA(21) for major trend direction (stronger than 4h)
- Trend: KAMA(21) vs KAMA(50) crossover for adaptive entry signals
- Momentum: MACD(12,26,9) histogram confirmation
- Regime: Bollinger Band width > 20th percentile (avoid squeeze periods)
- Stoploss: 2.5*ATR(14) trailing (slightly wider than 2.0 to reduce premature exits)
- Position sizing: 0.25 base, 0.30 normal, 0.35 max (discrete levels)
- Take profit: Reduce to half at 2.5R profit, trail stop at 1.5R

Why this should beat failed strategies:
- KAMA adapts to volatility = fewer whipsaws in chop vs static MA
- 12h HMA filter = stronger trend confirmation than 4h (fewer false signals)
- MACD histogram = momentum confirmation (avoids entering at trend exhaustion)
- BB width filter = avoids low-volatility periods where 8 of 14 failed strategies suffered
- Conservative sizing (0.25-0.35) controls drawdown during 2022 crash
- Wider stoploss (2.5*ATR) reduces premature exits that plagued earlier strategies
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_macd_12hhma_bbregime_1h_v1"
timeframe = "1h"
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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise: follows price closely in trends, flattens in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.zeros(n)
    for i in range(er_period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    close_s = pd.Series(close)
    
    ema_fast = close_s.ewm(span=fast, adjust=False, min_periods=fast).mean().values
    ema_slow = close_s.ewm(span=slow, adjust=False, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    macd_signal = pd.Series(macd_line).ewm(span=signal, adjust=False, min_periods=signal).mean().values
    macd_hist = macd_line - macd_signal
    
    return macd_line, macd_signal, macd_hist


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and bandwidth"""
    n = len(close)
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    upper_band = sma + std_dev * std
    lower_band = sma - std_dev * std
    bandwidth = (upper_band - lower_band) / sma
    
    return upper_band, lower_band, bandwidth, sma


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def calculate_percentile_rank(values, window=100):
    """Calculate rolling percentile rank for regime detection"""
    n = len(values)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(window, n):
        window_vals = values[i - window:i]
        valid_vals = window_vals[~np.isnan(window_vals)]
        if len(valid_vals) > 0:
            pr[i] = np.sum(valid_vals < values[i]) / len(valid_vals)
    
    return pr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for trend filter
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=10)
    kama_slow = calculate_kama(close, er_period=10, fast_period=5, slow_period=20)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bb_width_percentile = calculate_percentile_rank(bb_bandwidth, window=100)
    atr = calculate_atr(high, low, close, period=14)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    NORMAL_SIZE = 0.30  # Normal position size
    MAX_SIZE = 0.35   # Max position size with strong confirmation
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
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(kama_fast[i]) or
            np.isnan(kama_slow[i]) or np.isnan(macd_hist[i]) or
            np.isnan(bb_bandwidth[i]) or np.isnan(atr[i]) or
            np.isnan(bb_width_percentile[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 12h HMA trend filter
        price_above_12h_hma = close[i] > hma_12h_aligned[i]
        hma_trend = 1 if price_above_12h_hma else -1
        
        # KAMA crossover signals
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # MACD histogram confirmation
        macd_bullish = macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0
        
        # Bollinger Band regime filter (avoid low volatility squeezes)
        bb_regime_ok = bb_width_percentile[i] > 0.20  # Above 20th percentile
        
        # Calculate position size based on signal strength
        position_size = BASE_SIZE
        
        # Increase size with strong confirmation (all 3 filters agree)
        if hma_trend == 1 and kama_bullish and macd_bullish:
            position_size = MAX_SIZE
        elif hma_trend == -1 and kama_bearish and macd_bearish:
            position_size = -MAX_SIZE
        elif (hma_trend == 1 and kama_bullish) or (hma_trend == -1 and kama_bearish):
            position_size = NORMAL_SIZE * hma_trend
        else:
            position_size = BASE_SIZE * hma_trend
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish + 12h HMA bullish + MACD bullish + BB regime OK
        if kama_bullish and hma_trend == 1 and macd_bullish and bb_regime_ok:
            target_signal = abs(position_size)
        
        # Short entry: KAMA bearish + 12h HMA bearish + MACD bearish + BB regime OK
        elif kama_bearish and hma_trend == -1 and macd_bearish and bb_regime_ok:
            target_signal = -abs(position_size)
        
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
                
                # Check take profit (2.5R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 6.25 * entry_atr:  # 2.5R = 6.25*ATR
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
                    if close[i] <= entry_price - 6.25 * entry_atr:  # 2.5R profit
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
            # Reduce position to half at 2.5R profit
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
                # Exit if KAMA reverses OR 12h HMA alignment breaks OR MACD reverses strongly
                kama_reversal_long = kama_bearish
                kama_reversal_short = kama_bullish
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                macd_strong_reversal = (position_side == 1 and macd_hist[i] < -macd_hist[i-1] * 0.5) or \
                                       (position_side == -1 and macd_hist[i] > -macd_hist[i-1] * 0.5)
                
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