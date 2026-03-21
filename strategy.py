#!/usr/bin/env python3
"""
EXPERIMENT #007 - MACD Momentum + Bollinger Regime + 4h HMA Trend (15m primary)
=====================================================================================
Hypothesis: MACD histogram captures momentum shifts earlier than Supertrend. 
Bollinger Band Width detects regime (squeeze = breakout coming, wide = trend ongoing).
4h HMA(21) filters trades to only go with higher timeframe trend direction.
Volume confirmation ensures we're not trading low-liquidity fakeouts.

Key features:
- Primary TF: 15m
- HTF filter: 4h HMA(21) for major trend direction
- Momentum: MACD(12,26,9) histogram crossing zero
- Regime: Bollinger Band Width percentile (squeeze vs expansion)
- Entry: MACD cross + BB regime + volume spike + HTF alignment
- Stoploss: 2.0*ATR(14) trailing stop
- Position sizing: 0.20-0.30 discrete levels (conservative to control DD)
- Take profit: Reduce to half at 2R, trail stop at 1R

Why this should beat previous attempts:
- MACD histogram leads price more than Supertrend (earlier entries)
- BB regime filter avoids entering during chop (low BBW = wait for breakout)
- Volume confirmation filters low-liquidity false signals
- Conservative 0.25 base size prevents -50%+ drawdowns seen in #001, #003
- Discrete signal levels (0.0, ±0.25, ±0.30) minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "macd_bb_4hhma_15m_v1"
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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_s.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    band_width = (upper - lower) / sma
    return upper.values, lower.values, band_width.values


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def calculate_volume_spike(volume, period=20):
    """Detect volume spikes relative to recent average"""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_std = vol_s.rolling(window=period, min_periods=period).std()
    vol_zscore = (volume - vol_avg) / (vol_std + 1e-8)
    return vol_zscore


def calculate_bb_percentile(band_width, lookback=100):
    """Calculate where current BBW sits relative to recent history (percentile)"""
    n = len(band_width)
    bb_percentile = np.zeros(n)
    bb_percentile[:] = np.nan
    
    for i in range(lookback, n):
        window = band_width[i-lookback:i+1]
        if len(window) > 0 and not np.all(np.isnan(window)):
            current = band_width[i]
            percentile = np.nanpercentile(window, 100) 
            if percentile > 0:
                bb_percentile[i] = (current - np.nanmin(window)) / (np.nanmax(window) - np.nanmin(window) + 1e-8) * 100
            else:
                bb_percentile[i] = 50.0
    
    return bb_percentile


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
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    atr = calculate_atr(high, low, close, period=14)
    vol_zscore = calculate_volume_spike(volume, period=20)
    bb_percentile = calculate_bb_percentile(bb_width, lookback=100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital - conservative)
    MAX_SIZE = 0.30   # Max position size with strong confirmation
    MIN_SIZE = 0.15   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(macd_hist[i]) or
            np.isnan(bb_width[i]) or np.isnan(atr[i]) or np.isnan(vol_zscore[i]) or
            np.isnan(bb_percentile[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend filter
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_trend = 1 if price_above_4h_hma else -1
        
        # MACD momentum signals
        macd_bullish_cross = macd_hist[i] > 0 and macd_hist[i-1] <= 0  # Histogram crossing above zero
        macd_bearish_cross = macd_hist[i] < 0 and macd_hist[i-1] >= 0  # Histogram crossing below zero
        macd_momentum_long = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1]  # Positive and increasing
        macd_momentum_short = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1]  # Negative and decreasing
        
        # Bollinger Band regime
        bb_squeeze = bb_percentile[i] < 30  # Low percentile = squeeze (breakout coming)
        bb_expansion = bb_percentile[i] > 70  # High percentile = trend ongoing
        bb_breakout_long = close[i] > bb_upper[i] and bb_expansion
        bb_breakout_short = close[i] < bb_lower[i] and bb_expansion
        
        # Volume confirmation
        volume_confirmed = vol_zscore[i] > 0.5  # Above average volume
        
        # Calculate position size based on regime strength
        if bb_expansion and volume_confirmed:
            position_size = MAX_SIZE
        elif bb_squeeze or volume_confirmed:
            position_size = BASE_SIZE
        else:
            position_size = MIN_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: MACD momentum + 4h HMA bullish + BB regime + volume
        # Two entry modes: (1) MACD cross in expansion, (2) BB breakout with momentum
        long_condition_1 = (macd_bullish_cross or macd_momentum_long) and hma_trend == 1 and volume_confirmed
        long_condition_2 = bb_breakout_long and hma_trend == 1 and macd_momentum_long
        
        if long_condition_1 or long_condition_2:
            target_signal = position_size
        
        # Short entry: MACD momentum + 4h HMA bearish + BB regime + volume
        short_condition_1 = (macd_bearish_cross or macd_momentum_short) and hma_trend == -1 and volume_confirmed
        short_condition_2 = bb_breakout_short and hma_trend == -1 and macd_momentum_short
        
        if short_condition_1 or short_condition_2:
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
                # Exit if MACD reverses OR 4h HMA alignment breaks
                macd_reversal_long = macd_hist[i] < 0 and macd_hist[i-1] >= 0
                macd_reversal_short = macd_hist[i] > 0 and macd_hist[i-1] <= 0
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
                if macd_reversal_long or macd_reversal_short or hma_alignment_broken:
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