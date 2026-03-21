#!/usr/bin/env python3
"""
EXPERIMENT #008 - MACD Histogram + Volume + 4h HMA Trend (30m primary)
=====================================================================================
Hypothesis: MACD histogram captures momentum shifts earlier than signal line crossovers.
Combined with volume confirmation (volume > 1.5x average) and 4h HMA trend filter,
this should catch real breakouts while filtering false signals. 30m timeframe provides
good balance between signal frequency and noise reduction.

Key features:
- Primary TF: 30m
- HTF filter: 4h HMA(21) for major trend direction
- Entry: MACD histogram turning positive/negative with volume confirmation
- Strength: Volume > 1.5x 20-period average
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit, trail stop

Why this should work:
- MACD histogram leads signal line crossovers (earlier entry)
- Volume filter ensures we trade with real market interest
- 4h HMA keeps us with major trend (reduces whipsaws)
- 30m captures more opportunities than 1h/4h strategies
- Conservative sizing controls drawdown during crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "macd_vol_4hhma_30m_v1"
timeframe = "30m"
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
    """Calculate MACD indicator (histogram, macd line, signal line)"""
    n = len(close)
    close_s = pd.Series(close)
    
    ema_fast = close_s.ewm(span=fast, adjust=False, min_periods=fast).mean().values
    ema_slow = close_s.ewm(span=slow, adjust=False, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return histogram, macd_line, signal_line


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


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
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average"""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    volume_ratio = volume / vol_ma
    volume_ratio[np.isnan(volume_ratio)] = 1.0
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
    
    # Calculate 30m indicators
    macd_hist, macd_line, macd_signal = calculate_macd(close, fast=12, slow=26, signal=9)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    volume_ratio = calculate_volume_ratio(volume, period=20)
    
    # Also calculate 4h HMA slope for additional filter
    hma_4h_shifted = np.roll(hma_4h_aligned, 1)
    hma_4h_shifted[:50] = np.nan
    hma_slope = hma_4h_aligned - hma_4h_shifted
    
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
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(macd_hist[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(volume_ratio[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend filter
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_trend = 1 if price_above_4h_hma else -1
        
        # 4h HMA slope (trend strengthening)
        hma_slope_positive = hma_slope[i] > 0 if not np.isnan(hma_slope[i]) else False
        hma_slope_negative = hma_slope[i] < 0 if not np.isnan(hma_slope[i]) else False
        
        # MACD histogram signals (momentum shift)
        macd_bullish = macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0
        
        # MACD histogram turning (momentum acceleration)
        macd_turning_long = macd_hist[i] > 0 and macd_hist[i-1] <= 0 if i > 0 else False
        macd_turning_short = macd_hist[i] < 0 and macd_hist[i-1] >= 0 if i > 0 else False
        
        # Volume confirmation
        volume_strong = volume_ratio[i] > 1.3  # Volume 30% above average
        
        # RSI filter (avoid extreme overbought/oversold for entries)
        rsi_not_overbought = rsi[i] < 75
        rsi_not_oversold = rsi[i] > 25
        
        # Calculate position size based on volume strength (dynamic sizing)
        vol_multiplier = min(1.0 + (volume_ratio[i] - 1.3) / 2.0, 1.25)  # Max 1.25x
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * vol_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: MACD bullish + 4h HMA bullish + Volume strong + RSI not overbought
        # Allow entry on MACD turning OR sustained bullish with volume
        if (macd_bullish and hma_trend == 1 and rsi_not_overbought):
            if volume_strong or macd_turning_long:
                target_signal = position_size
        
        # Short entry: MACD bearish + 4h HMA bearish + Volume strong + RSI not oversold
        elif (macd_bearish and hma_trend == -1 and rsi_not_oversold):
            if volume_strong or macd_turning_short:
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
                # Exit if MACD reverses OR 4h HMA alignment breaks
                macd_reversal_long = macd_bearish
                macd_reversal_short = macd_bullish
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
                # Also exit on extreme RSI (potential reversal)
                rsi_extreme_long = rsi[i] > 80
                rsi_extreme_short = rsi[i] < 20
                
                if macd_reversal_long or macd_reversal_short or hma_alignment_broken or \
                   rsi_extreme_long or rsi_extreme_short:
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