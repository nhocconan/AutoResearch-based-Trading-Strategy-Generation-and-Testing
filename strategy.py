#!/usr/bin/env python3
"""
EXPERIMENT #001 - KAMA Trend + Bollinger Squeeze + 4h HMA Filter (15m primary)
=====================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than EMA/HMA,
reducing whipsaws in choppy markets. Bollinger Band Width detects squeeze periods that
precede explosive moves. Combining KAMA crossover signals with squeeze breakouts and
4h HMA trend filter should capture high-probability momentum moves while avoiding chop.

Key features:
- Primary TF: 15m
- HTF filter: 4h HMA(21) for major trend direction
- Trend: KAMA(10) vs KAMA(40) crossover (adaptive to volatility)
- Regime: Bollinger Band Width < 20th percentile (squeeze detection)
- Entry: KAMA crossover + squeeze breakout + HTF alignment + volume confirmation
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25 base, 0.35 max with strong trend
- Take profit: Reduce to half at 2.5R profit

Why this should beat current best:
- KAMA reduces false signals in chop vs Supertrend
- Bollinger squeeze filter captures explosive moves after consolidation
- 4h HMA ensures we trade with major trend
- Volume confirmation filters false breakouts
- Conservative sizing controls drawdown during crypto crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_bb_squeeze_4hhma_15m_v1"
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


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market volatility - moves fast in trends, slow in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    upper_band = sma + std_dev * std
    lower_band = sma - std_dev * std
    band_width = (upper_band - lower_band) / sma  # Normalized bandwidth
    
    return upper_band, lower_band, sma, band_width


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend filter
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    kama_fast = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, period=40, fast_period=2, slow_period=30)
    
    bb_upper, bb_lower, bb_sma, bb_width = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    atr = calculate_atr(high, low, close, period=14)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    # Calculate Bollinger Width percentile for squeeze detection
    bb_width_s = pd.Series(bb_width)
    bb_width_percentile = bb_width_s.rolling(window=100, min_periods=50).apply(
        lambda x: (x < x.iloc[-1]).mean() * 100 if len(x) > 0 else 50
    ).values
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size with strong trend
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
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(kama_fast[i]) or
            np.isnan(kama_slow[i]) or np.isnan(bb_width[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend filter
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_trend = 1 if price_above_4h_hma else -1
        
        # KAMA crossover signals
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # KAMA crossover detection (fast crosses above/below slow)
        kama_cross_long = (kama_fast[i] > kama_slow[i] and 
                          kama_fast[i-1] <= kama_slow[i-1])
        kama_cross_short = (kama_fast[i] < kama_slow[i] and 
                           kama_fast[i-1] >= kama_slow[i-1])
        
        # Bollinger squeeze detection (width in bottom 20%)
        squeeze_active = bb_width_percentile[i] < 20 if not np.isnan(bb_width_percentile[i]) else False
        
        # Volume confirmation (volume > 1.2x average)
        volume_confirmation = volume[i] > 1.2 * vol_ma[i] if vol_ma[i] > 0 else False
        
        # Price above/below BB middle for momentum confirmation
        price_above_bb_mid = close[i] > bb_sma[i]
        price_below_bb_mid = close[i] < bb_sma[i]
        
        # Calculate position size based on trend strength
        trend_strength = 1.0
        if hma_trend == 1 and kama_bullish:
            trend_strength = 1.15  # Strong uptrend
        elif hma_trend == -1 and kama_bearish:
            trend_strength = 1.15  # Strong downtrend
        
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * trend_strength))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish + 4h HMA bullish + squeeze or volume confirmation + price above BB mid
        if (kama_bullish and hma_trend == 1 and price_above_bb_mid and
            (squeeze_active or volume_confirmation)):
            # Extra confirmation on crossover
            if kama_cross_long or position_side == 1:
                target_signal = position_size
        
        # Short entry: KAMA bearish + 4h HMA bearish + squeeze or volume confirmation + price below BB mid
        elif (kama_bearish and hma_trend == -1 and price_below_bb_mid and
              (squeeze_active or volume_confirmation)):
            # Extra confirmation on crossover
            if kama_cross_short or position_side == -1:
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