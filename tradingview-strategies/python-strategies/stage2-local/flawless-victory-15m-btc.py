#!/usr/bin/env python3
"""
Flawless Victory Strategy - 15min BTC Machine Learning Strategy
Converted from TradingView Pine Script (Trebor_Namor)

Note: This is a partial conversion. Pine strategy.exit with stop/limit
levels are approximated as next-bar close signals based on entry price.
"""

import numpy as np
import pandas as pd

name = "Flawless Victory Strategy - 15min BTC"
timeframe = "15m"
leverage = 1

# Strategy version parameters (default to v1)
VERSION = 1  # 1, 2, or 3
V2_STOP_LOSS_PCT = 6.604 / 100
V2_TAKE_PROFIT_PCT = 2.328 / 100
V3_STOP_LOSS_PCT = 8.882 / 100
V3_TAKE_PROFIT_PCT = 2.317 / 100


def _rma(series, length):
    """Calculate RMA (Running Moving Average) similar to Pine Script."""
    result = np.zeros(len(series))
    alpha = 1.0 / length
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def _calculate_rsi(close, length=14):
    """Calculate RSI using RMA for up/down moves."""
    change = np.diff(close, prepend=close[0])
    up = np.maximum(change, 0)
    down = -np.minimum(change, 0)
    
    up_rma = _rma(up, length)
    down_rma = _rma(down, length)
    
    rsi = np.zeros(len(close))
    for i in range(len(close)):
        if down_rma[i] == 0:
            rsi[i] = 100
        elif up_rma[i] == 0:
            rsi[i] = 0
        else:
            rsi[i] = 100 - 100 / (1 + up_rma[i] / down_rma[i])
    
    return rsi


def _calculate_mfi(high, low, close, volume, length=14):
    """Calculate Money Flow Index similar to Pine Script."""
    hlc3 = (high + low + close) / 3.0
    change = np.diff(hlc3, prepend=hlc3[0])
    
    positive_flow = np.where(change > 0, volume * hlc3, 0)
    negative_flow = np.where(change < 0, volume * hlc3, 0)
    
    # Use simple sum for MFI (Pine uses sum, not RMA)
    upper = np.zeros(len(hlc3))
    lower = np.zeros(len(hlc3))
    
    for i in range(len(hlc3)):
        start_idx = max(0, i - length + 1)
        upper[i] = np.sum(positive_flow[start_idx:i + 1])
        lower[i] = np.sum(negative_flow[start_idx:i + 1])
    
    mfi = np.zeros(len(hlc3))
    for i in range(len(hlc3)):
        if lower[i] == 0:
            mfi[i] = 100
        elif upper[i] == 0:
            mfi[i] = 0
        else:
            mfi[i] = 100.0 - (100.0 / (1.0 + upper[i] / lower[i]))
    
    return mfi


def _calculate_bollinger_bands(close, length, mult=1.0):
    """Calculate Bollinger Bands."""
    basis = pd.Series(close).rolling(window=length, min_periods=1).mean().values
    std = pd.Series(close).rolling(window=length, min_periods=1).std().values
    std = np.nan_to_num(std, nan=0.0)
    
    upper = basis + mult * std
    lower = basis - mult * std
    
    return basis, upper, lower


def generate_signals(prices):
    """
    Generate target position signals based on Flawless Victory Strategy logic.
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume]
        
    Returns:
        numpy.ndarray with target position fractions (-1 to 1) for each bar.
        Signals represent target position at next bar open.
    """
    n = len(prices)
    if n == 0:
        return np.array([])
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate indicators
    rsi = _calculate_rsi(close, 14)
    mfi = _calculate_mfi(high, low, close, volume, 14)
    
    # Version-specific Bollinger Bands
    if VERSION == 2:
        bb_length = 17
    else:
        bb_length = 20
    
    bb_basis, bb_upper, bb_lower = _calculate_bollinger_bands(close, bb_length, 1.0)
    
    # Initialize signals array (0 = no position, 1 = long)
    signals = np.zeros(n)
    
    # Track position state for SL/TP logic
    in_position = False
    entry_price = 0.0
    
    # Version-specific parameters
    if VERSION == 1:
        rsi_buy_level = 42
        rsi_sell_level = 70
    elif VERSION == 2:
        rsi_buy_level = 42
        rsi_sell_level = 76
    else:  # VERSION == 3
        mfi_buy_level = 60
        rsi_sell_level = 65
        mfi_sell_level = 64
    
    for i in range(n):
        # Skip warmup period for indicators
        if i < 20:
            signals[i] = 0
            continue
        
        # Entry conditions (price below lower BB)
        bb_buy_trigger = close[i] < bb_lower[i]
        bb_sell_trigger = close[i] > bb_upper[i]
        
        if VERSION == 1:
            buy_signal = bb_buy_trigger and (rsi[i] > rsi_buy_level)
            sell_signal = bb_sell_trigger and (rsi[i] > rsi_sell_level)
        elif VERSION == 2:
            buy_signal = bb_buy_trigger and (rsi[i] > rsi_buy_level)
            sell_signal = bb_sell_trigger and (rsi[i] > rsi_sell_level)
        else:  # VERSION == 3
            buy_signal = bb_buy_trigger and (mfi[i] < mfi_buy_level)
            sell_signal = bb_sell_trigger and (rsi[i] > rsi_sell_level) and (mfi[i] > mfi_sell_level)
        
        # Handle position state and SL/TP for v2/v3
        if VERSION in [2, 3]:
            if VERSION == 2:
                sl_pct = V2_STOP_LOSS_PCT
                tp_pct = V2_TAKE_PROFIT_PCT
            else:
                sl_pct = V3_STOP_LOSS_PCT
                tp_pct = V3_TAKE_PROFIT_PCT
            
            # Check SL/TP before new signals (next-bar execution)
            if in_position and entry_price > 0:
                sl_level = entry_price * (1 - sl_pct)
                tp_level = entry_price * (1 + tp_pct)
                
                # Check if price triggered SL or TP (using low/high for approximation)
                if low[i] <= sl_level or high[i] >= tp_level:
                    signals[i] = 0  # Exit position
                    in_position = False
                    entry_price = 0.0
                    continue
        
        # Process entry/exit signals (next-bar execution)
        if buy_signal and not in_position:
            signals[i] = 1  # Enter long at next bar open
            in_position = True
            entry_price = close[i]  # Track entry for SL/TP
        elif sell_signal and in_position:
            signals[i] = 0  # Exit at next bar open
            in_position = False
            entry_price = 0.0
        elif in_position:
            signals[i] = 1  # Maintain position
        else:
            signals[i] = 0  # No position
    
    return signals
