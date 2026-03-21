#!/usr/bin/env python3
"""
EXPERIMENT #002 - KAMA Adaptive Trend + Bollinger Regime + 4h HMA Filter (30m)
==============================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency,
performing better than fixed EMAs in varying volatility regimes. Combined with
Bollinger Band Width to detect low-volatility squeezes (avoid choppy markets)
and 4h HMA for higher timeframe trend alignment. RSI pullback entries within
the trend direction improve entry timing vs pure breakouts.

Key features:
- Primary TF: 30m (faster signals than 1h/4h, slower than 5m/15m)
- HTF filter: 4h HMA(21) for major trend direction
- Trend: KAMA(10) vs KAMA(40) crossover with efficiency ratio filter
- Regime: Bollinger Band Width percentile (avoid bottom 30% = choppy)
- Entry: RSI(14) pullback to 40-60 zone within trend
- Stoploss: 2.0*ATR(14) trailing
- Take profit: Reduce to half at 2R, trail stop at 1R
- Position sizing: 0.25-0.30 discrete levels

Why different from failed #001:
- KAMA adapts to volatility vs fixed Supertrend parameters
- Bollinger BW regime filter avoids choppy markets (major failure cause)
- RSI pullback entries vs RSI extreme entries (better risk/reward)
- 30m timeframe captures more intraday moves than 15m
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_bollinger_regime_4h_30m_v1"
timeframe = "30m"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency ratio
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(slow_period, n):
        price_change = abs(close[i] - close[i - slow_period])
        volatility = np.sum(np.abs(np.diff(close[i - slow_period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Initialize KAMA
    kama[slow_period] = close[slow_period]
    
    for i in range(slow_period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
            continue
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
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


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    band_width = (upper - lower) / sma
    return upper.values, lower.values, band_width.values


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


def calculate_bb_width_percentile(band_width, lookback=100):
    """Calculate rolling percentile of Bollinger Band Width"""
    n = len(band_width)
    percentile = np.zeros(n)
    percentile[:] = np.nan
    
    for i in range(lookback, n):
        window = band_width[i - lookback:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            percentile[i] = np.sum(valid <= band_width[i]) / len(valid) * 100
        else:
            percentile[i] = 50.0
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    kama_fast = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, period=40, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, 100)
    
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
    
    min_period = 150  # Wait for KAMA, BB, and 4h HMA to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(kama_fast[i]) or 
            np.isnan(kama_slow[i]) or np.isnan(atr[i]) or 
            np.isnan(rsi[i]) or np.isnan(bb_width_pct[i]) or 
            atr[i] == 0 or bb_width[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter (HTF alignment)
        htf_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # Bollinger Band Width regime filter (avoid choppy markets)
        # Only trade when BB width is in top 70% (not in squeeze)
        regime_valid = bb_width_pct[i] >= 30.0
        
        # KAMA trend direction
        kama_trend = 0
        if kama_fast[i] > kama_slow[i]:
            kama_trend = 1
        elif kama_fast[i] < kama_slow[i]:
            kama_trend = -1
        
        # KAMA efficiency filter (fast KAMA should be moving)
        kama_moving = False
        if i >= 5:
            kama_change = abs(kama_fast[i] - kama_fast[i - 5]) / kama_fast[i - 5] if kama_fast[i - 5] > 0 else 0
            kama_moving = kama_change > 0.001  # At least 0.1% move in 5 bars
        
        # RSI pullback entry signal (within trend, not extreme)
        rsi_signal = 0
        if htf_trend == 1 and kama_trend == 1:
            # Long: RSI pullback to 40-55 zone
            if 40 <= rsi[i] <= 55:
                rsi_signal = 1
        elif htf_trend == -1 and kama_trend == -1:
            # Short: RSI pullback to 45-60 zone
            if 45 <= rsi[i] <= 60:
                rsi_signal = -1
        
        # Determine target signal based on all filters
        target_signal = 0.0
        if rsi_signal != 0 and regime_valid and kama_moving:
            # All filters aligned
            target_signal = SIZE * rsi_signal
        
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
                
                # Check take profit (2R from entry, R = 2*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * atr[i]:  # 2R = 4*ATR
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
                    if close[i] <= entry_price - 4.0 * atr[i]:  # 2R profit
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
            # Trail stop tighter after TP (1R from highest/lowest)
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, close[i])
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            # Apply signal change
            if target_signal != 0.0:
                signals[i] = target_signal
                if position_side == 0:
                    # New entry
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
                    profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals