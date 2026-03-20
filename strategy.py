#!/usr/bin/env python3
"""
EXPERIMENT #006 - Multi-Timeframe Supertrend + RSI Pullback + Volume Filter
=============================================================================
Hypothesis: 4h Supertrend provides robust trend direction, 1h RSI pullback
gives precise entry timing during trend continuations. Volume confirmation
filters out false breakouts. ATR trailing stop reduces drawdown.

Key improvements over mtf_dema_macd_bbregime_v1:
- Supertrend is more robust than DEMA for trend detection (proven in exp#001)
- RSI pullback entries are more reliable than MACD histogram crosses
- Volume spike confirmation reduces false signals
- Fixed read-only array issues by using .copy() properly
- Cleaner stoploss tracking with explicit entry price management

Why this might beat Sharpe=1.768:
- Supertrend + RSI combo has worked well historically
- Volume filter adds extra confirmation layer
- Better signal transition logic reduces churn costs
- Proper position sizing (0.20-0.35) controls drawdown
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_rsi_volume_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    n = len(close)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator
    Returns: supertrend_line, trend_direction (1=up, -1=down)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
            
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = -1
        else:
            # Trend logic
            if close[i-1] <= supertrend[i-1]:
                # Previously in downtrend
                if close[i] > upper_band[i]:
                    supertrend[i] = lower_band[i]
                    trend[i] = 1
                else:
                    supertrend[i] = min(upper_band[i], supertrend[i-1])
                    trend[i] = -1
            else:
                # Previously in uptrend
                if close[i] < lower_band[i]:
                    supertrend[i] = upper_band[i]
                    trend[i] = -1
                else:
                    supertrend[i] = max(lower_band[i], supertrend[i-1])
                    trend[i] = 1
    
    # Set initial values to nan
    supertrend[:period] = np.nan
    trend[:period] = 0
    
    return supertrend, trend


def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    
    return rsi


def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above average"""
    n = len(volume)
    avg_volume = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    volume_ratio = np.zeros(n)
    mask = avg_volume > 0
    volume_ratio[mask] = volume[mask] / avg_volume[mask]
    
    spike = np.zeros(n)
    spike[volume_ratio > threshold] = 1.0
    
    return spike, volume_ratio


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices.get("volume", np.ones(len(close))).values.copy()
    n = len(close)
    
    # ========== 1h Indicators for Entry Timing ==========
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    volume_spike_1h, volume_ratio_1h = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    # ========== 4h Trend Filter (resample 1h → 4h) ==========
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    close_4h = df_4h['close'].values.copy()
    high_4h = df_4h['high'].values.copy()
    low_4h = df_4h['low'].values.copy()
    n_4h = len(close_4h)
    
    # 4h Supertrend for trend direction
    supertrend_4h, trend_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
    
    # 4h ATR for stoploss (scaled to 1h)
    atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    atr_1h_mapped = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
        if idx_4h < len(atr_4h) and not np.isnan(atr_4h[idx_4h]):
            atr_1h_mapped[i] = atr_4h[idx_4h] / 2.0  # Scale 4h ATR to 1h
    
    # ========== Generate Signals ==========
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels
    SIZE_FULL = 0.35
    SIZE_HALF = 0.18
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45.0   # Buy pullback in uptrend
    RSI_SHORT_ENTRY = 55.0  # Sell pullback in downtrend
    RSI_EXIT = 50.0         # Neutral exit point
    
    # Volume confirmation
    VOLUME_CONFIRM = True   # Require volume spike for entries
    
    # Wait for all indicators to be valid
    first_valid = max(14, 10, 20, 100)  # RSI, Supertrend, Volume, mapping
    
    # Track entry prices for stoploss
    entry_price = np.zeros(n)
    position_direction = np.zeros(n)  # 1=long, -1=short, 0=none
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Check for valid data
        if (np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or
            np.isnan(trend_1h[i]) or np.isnan(atr_1h_mapped[i]) or
            atr_1h_mapped[i] == 0):
            signals[i] = 0.0
            position_direction[i] = 0
            continue
        
        trend = trend_1h[i]
        rsi = rsi_1h[i]
        atr_val = atr_1h_mapped[i]
        vol_spike = volume_spike_1h[i]
        
        # Initialize tracking arrays
        if i > 0:
            entry_price[i] = entry_price[i-1]
            position_direction[i] = position_direction[i-1]
            highest_since_entry[i] = highest_since_entry[i-1]
            lowest_since_entry[i] = lowest_since_entry[i-1]
        
        # Check ATR trailing stoploss (2*ATR against position)
        if position_direction[i] != 0 and entry_price[i] > 0:
            if position_direction[i] == 1:  # Long position
                # Update highest since entry for trailing
                if close[i] > highest_since_entry[i]:
                    highest_since_entry[i] = close[i]
                
                stop_loss = entry_price[i] - 2 * atr_val
                trailing_stop = highest_since_entry[i] - 2 * atr_val
                effective_stop = max(stop_loss, trailing_stop)
                
                if close[i] < effective_stop:
                    signals[i] = 0.0
                    position_direction[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    continue
            elif position_direction[i] == -1:  # Short position
                # Update lowest since entry for trailing
                if close[i] < lowest_since_entry[i] or lowest_since_entry[i] == 0:
                    lowest_since_entry[i] = close[i]
                
                stop_loss = entry_price[i] + 2 * atr_val
                trailing_stop = lowest_since_entry[i] + 2 * atr_val
                effective_stop = min(stop_loss, trailing_stop)
                
                if close[i] > effective_stop:
                    signals[i] = 0.0
                    position_direction[i] = 0
                    entry_price[i] = 0
                    lowest_since_entry[i] = 0
                    continue
        
        # Generate new signals based on trend + RSI pullback
        if trend == 1:  # 4h uptrend - look for long entries
            if rsi < RSI_LONG_ENTRY:
                # RSI pullback - potential long entry
                if position_direction[i] == 0:
                    # New entry - check volume confirmation
                    if VOLUME_CONFIRM and vol_spike < 1.0:
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE_FULL
                        position_direction[i] = 1
                        entry_price[i] = close[i]
                        highest_since_entry[i] = close[i]
                elif position_direction[i] == 1:
                    # Already long - maintain or add
                    signals[i] = SIZE_FULL
            elif rsi > 65.0 and position_direction[i] == 1:
                # RSI overbought - reduce position (take profit)
                signals[i] = SIZE_HALF
            elif rsi > 70.0 and position_direction[i] == 1:
                # RSI very overbought - exit
                signals[i] = 0.0
                position_direction[i] = 0
                entry_price[i] = 0
                highest_since_entry[i] = 0
            else:
                if position_direction[i] == 1:
                    signals[i] = SIZE_FULL
                else:
                    signals[i] = 0.0
                
        elif trend == -1:  # 4h downtrend - look for short entries
            if rsi > RSI_SHORT_ENTRY:
                # RSI pullback - potential short entry
                if position_direction[i] == 0:
                    # New entry - check volume confirmation
                    if VOLUME_CONFIRM and vol_spike < 1.0:
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE_FULL
                        position_direction[i] = -1
                        entry_price[i] = close[i]
                        lowest_since_entry[i] = close[i]
                elif position_direction[i] == -1:
                    # Already short - maintain or add
                    signals[i] = -SIZE_FULL
            elif rsi < 35.0 and position_direction[i] == -1:
                # RSI oversold - reduce position (take profit)
                signals[i] = -SIZE_HALF
            elif rsi < 30.0 and position_direction[i] == -1:
                # RSI very oversold - exit
                signals[i] = 0.0
                position_direction[i] = 0
                entry_price[i] = 0
                lowest_since_entry[i] = 0
            else:
                if position_direction[i] == -1:
                    signals[i] = -SIZE_FULL
                else:
                    signals[i] = 0.0
        else:  # No clear trend
            signals[i] = 0.0
            position_direction[i] = 0
            entry_price[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
    
    return signals