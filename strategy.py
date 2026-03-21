#!/usr/bin/env python3
"""
EXPERIMENT #015 - MACD Momentum + HTF Trend + Volume Confirmation (1h primary, 12h HTF)
========================================================================================
Hypothesis: 1h MACD histogram captures momentum shifts better than RSI for entries.
12h HMA(50) provides strong trend filter without being too slow (like 1d).
Volume spikes confirm genuine breakouts vs fake moves. Bollinger Band Width regime
filter avoids low-volatility chop where MACD whipsaws. This differs from failed
MACD attempts by adding proper HTF alignment, volume confirmation, and regime filter.

Key features:
- Primary TF: 1h (this experiment's requirement)
- HTF filter: 12h HMA(50) for trend direction
- Momentum: MACD(12,26,9) histogram crossing zero with slope confirmation
- Volume: Volume > 1.5x 20-bar SMA confirms genuine moves
- Regime: BB Width > 40th percentile (avoid ultra-low vol chop)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels with take-profit reduction
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "macd_volume_htf_1h_12h_v1"
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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_s.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values


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


def calculate_bollinger_bands(close, period=20, std_dev=2):
    """Calculate Bollinger Bands and Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    band_width = (upper - lower) / sma
    return upper.values, lower.values, band_width.values


def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above threshold * SMA"""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    volume_spike = volume > (threshold * vol_sma)
    return volume_spike


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    hma_12h = calculate_hma(df_12h['close'].values, 50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    macd_line, signal_line, histogram = calculate_macd(close, 12, 26, 9)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2)
    volume_spike = calculate_volume_spike(volume, 20, 1.5)
    
    # Calculate MACD histogram slope (momentum acceleration)
    hist_slope = np.zeros(n)
    hist_slope[1:] = histogram[1:] - histogram[:-1]
    
    # Calculate BB Width percentile for regime filter
    bb_width_pr = np.zeros(n)
    bb_width_pr[:] = np.nan
    window = 100
    for i in range(window - 1, n):
        if not np.isnan(bb_width[i]):
            window_data = bb_width[i - window + 1:i + 1]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) > 0:
                bb_width_pr[i] = np.sum(window_data <= bb_width[i]) / len(window_data)
    
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
    
    min_period = 120  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(histogram[i]) or 
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(bb_width_pr[i]) or 
            atr[i] == 0 or np.isnan(hist_slope[i])):
            signals[i] = 0.0
            continue
        
        # 12h trend filter (HTF)
        htf_trend = 1 if close[i] > hma_12h_aligned[i] else -1
        
        # MACD momentum signals
        macd_bullish = histogram[i] > 0 and hist_slope[i] > 0
        macd_bearish = histogram[i] < 0 and hist_slope[i] < 0
        
        # MACD zero-cross confirmation (stronger signal)
        macd_cross_long = histogram[i] > 0 and histogram[i-1] <= 0
        macd_cross_short = histogram[i] < 0 and histogram[i-1] >= 0
        
        # Regime filter: only trade when BB Width is in top 60% (trending market)
        regime_valid = bb_width_pr[i] > 0.40
        
        # Volume confirmation
        vol_confirmed = volume_spike[i]
        
        # RSI filter to avoid overbought/oversold entries
        rsi_valid_long = rsi[i] < 70  # Not overbought
        rsi_valid_short = rsi[i] > 30  # Not oversold
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: HTF bullish + MACD bullish + Volume spike + Regime valid + RSI valid
        long_condition = (htf_trend == 1 and 
                         (macd_bullish or macd_cross_long) and 
                         vol_confirmed and 
                         regime_valid and 
                         rsi_valid_long)
        
        # Short entry: HTF bearish + MACD bearish + Volume spike + Regime valid + RSI valid
        short_condition = (htf_trend == -1 and 
                          (macd_bearish or macd_cross_short) and 
                          vol_confirmed and 
                          regime_valid and 
                          rsi_valid_short)
        
        if long_condition:
            target_signal = SIZE
        elif short_condition:
            target_signal = -SIZE
        
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
                
                # Check take profit (2R from entry, where R = 2.5*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * atr[i]:  # 2R = 5*ATR
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
                    if close[i] <= entry_price - 5.0 * atr[i]:  # 2R profit
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
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                if position_side == 1 and htf_trend == -1:
                    # HTF trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and htf_trend == 1:
                    # HTF trend reversed, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals