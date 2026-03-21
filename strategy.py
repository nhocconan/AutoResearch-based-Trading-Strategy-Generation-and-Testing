#!/usr/bin/env python3
"""
EXPERIMENT #037 - KAMA Adaptive Trend + BB Regime + 4h HMA Filter + Volume (15m primary)
=====================================================================================
Hypothesis: KAMA adapts to volatility better than EMA/HMA, reducing whipsaws in chop.
Combining with Bollinger Band Width regime detection ensures we only trade during
expansion phases (not squeezes). 4h HMA(21) provides higher timeframe trend filter.
Volume confirmation (1.5x average) ensures breakouts have participation.

Key features:
- Primary TF: 15m
- HTF filter: 4h HMA(21) for major trend direction
- Trend: KAMA(10, 2, 30) adaptive moving average crossover
- Regime: Bollinger Band Width percentile > 40 (expansion, not squeeze)
- Volume: Current volume > 1.5x 20-bar average
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.35 discrete levels
- Take profit: Reduce to half at 2.5R profit

Why this should beat current best:
- KAMA adapts to crypto volatility regimes better than fixed EMA
- BB regime filter avoids trading during low-volatility chop
- Volume filter reduces false breakout signals
- 15m captures more opportunities than 1h/4h strategies
- Conservative sizing controls drawdown during crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_bbregime_4hhma_vol_15m_v1"
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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - moves fast in trends, slow in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = 0.0
        for j in range(i - er_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(er_period, n):
        if i == er_period:
            kama[i] = close[i]
        else:
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    band_width = (upper - lower) / sma
    
    return upper, lower, sma, band_width


def calculate_bb_width_percentile(band_width, lookback=100):
    """Calculate percentile rank of BB Width over lookback period"""
    n = len(band_width)
    percentile = np.zeros(n)
    percentile[:] = np.nan
    
    for i in range(lookback, n):
        if np.isnan(band_width[i]):
            continue
        window = band_width[i - lookback + 1:i + 1]
        window = window[~np.isnan(window)]
        if len(window) > 0:
            percentile[i] = np.sum(window < band_width[i]) / len(window) * 100
    
    return percentile


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average"""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    volume_ratio = volume / vol_ma
    volume_ratio[vol_ma == 0] = 1.0
    return volume_ratio


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend filter
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=15)
    atr = calculate_atr(high, low, close, period=14)
    
    upper, lower, sma, band_width = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bb_percentile = calculate_bb_width_percentile(band_width, lookback=100)
    
    volume_ratio = calculate_volume_ratio(volume, period=20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size with strong signals
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 120  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(kama[i]) or np.isnan(kama_fast[i]) or
            np.isnan(atr[i]) or np.isnan(bb_percentile[i]) or np.isnan(volume_ratio[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend filter
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_trend = 1 if price_above_4h_hma else -1
        
        # KAMA trend direction (fast vs slow KAMA)
        kama_bullish = kama_fast[i] > kama[i]
        kama_bearish = kama_fast[i] < kama[i]
        
        # KAMA slope (price above/below KAMA)
        kama_slope_long = close[i] > kama[i]
        kama_slope_short = close[i] < kama[i]
        
        # BB regime filter (only trade in expansion, not squeeze)
        bb_expansion = bb_percentile[i] > 35  # Above 35th percentile = expansion
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] > 1.3  # 30% above average
        
        # Calculate position size based on signal strength
        signal_strength = 0
        if bb_expansion:
            signal_strength += 1
        if volume_confirmed:
            signal_strength += 1
        
        position_size = BASE_SIZE
        if signal_strength >= 2:
            position_size = MAX_SIZE
        elif signal_strength >= 1:
            position_size = (BASE_SIZE + MAX_SIZE) / 2
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish + 4h HMA bullish + BB expansion + volume confirmed
        if (kama_bullish and kama_slope_long and hma_trend == 1 and 
            bb_expansion and volume_confirmed):
            target_signal = position_size
        
        # Short entry: KAMA bearish + 4h HMA bearish + BB expansion + volume confirmed
        elif (kama_bearish and kama_slope_short and hma_trend == -1 and 
              bb_expansion and volume_confirmed):
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
                # Exit if KAMA reverses OR 4h HMA alignment breaks
                kama_reversal_long = kama_bearish
                kama_reversal_short = kama_bullish
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
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