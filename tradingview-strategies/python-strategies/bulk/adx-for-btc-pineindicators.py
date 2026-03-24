#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "ADX for BTC [PineIndicators]"
timeframe = "1h"
leverage = 1

def rma(series, length):
    """Calculates Wilder's Running Moving Average."""
    result = np.zeros_like(series, dtype=float)
    if len(series) < length:
        return result
    # First value is SMA of the first 'length' items
    result[length-1] = np.mean(series[:length])
    # Subsequent values use RMA formula
    for i in range(length, len(series)):
        result[i] = (result[i-1] * (length - 1) + series[i]) / length
    # Fill initial period with NaN
    result[:length-1] = np.nan
    return result

def calculate_adx(high, low, close, dilen=14, adxlen=14):
    """Calculates ADX indicator replicating Pine Script ta.adx logic."""
    high = np.array(high, dtype=float)
    low = np.array(low, dtype=float)
    close = np.array(close, dtype=float)
    
    # Directional Movement
    up = np.diff(high)
    down = -np.diff(low)
    
    # Pad to match original length (first bar is 0/na)
    up = np.concatenate([[0], up])
    down = np.concatenate([[0], down])
    
    plusDM = np.where((up > down) & (up > 0), up, 0.0)
    minusDM = np.where((down > up) & (down > 0), down, 0.0)
    
    # True Range
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values using RMA
    tr_smooth = rma(tr, dilen)
    plus_smooth = rma(plusDM, dilen)
    minus_smooth = rma(minusDM, dilen)
    
    # Directional Indicators
    with np.errstate(divide='ignore', invalid='ignore'):
        plusDI = 100 * plus_smooth / tr_smooth
        minusDI = 100 * minus_smooth / tr_smooth
        # Handle NaNs from division
        plusDI = np.nan_to_num(plusDI)
        minusDI = np.nan_to_num(minusDI)
        
        sumDI = plusDI + minusDI
        diffDI = np.abs(plusDI - minusDI)
        adx_raw = 100 * diffDI / np.where(sumDI == 0, 1, sumDI)
    
    adx = rma(adx_raw, adxlen)
    return adx

def generate_signals(prices):
    """
    Generates trading signals based on ADX crossover and SMA filter.
    
    Args:
        prices (pd.DataFrame or dict): Must contain 'open', 'high', 'low', 'close', 'volume'.
        
    Returns:
        np.ndarray: Array of position targets (0.0 for flat, 1.0 for long) matching input length.
    """
    if isinstance(prices, dict):
        df = pd.DataFrame(prices)
    else:
        df = prices.copy()
    
    # Strategy Parameters
    TP = 14.0       # Entry Level (ADX crossover threshold)
    SL = 45.0       # Exit Level (ADX crossunder threshold)
    SMA_LEN = 200   # SMA Filter Length
    SMA_FILTER = True
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate Indicators
    adx = calculate_adx(high, low, close)
    
    sma_short = pd.Series(close).rolling(window=SMA_LEN).mean().values
    sma_long = pd.Series(close).rolling(window=SMA_LEN * 5).mean().values
    
    # Signal Conditions
    # Shift ADX by 1 to compare current vs previous for crossover/crossunder
    adx_prev = np.concatenate([[np.nan], adx[:-1]])
    
    # Crossover: Current > TP and Previous <= TP
    crossover = (adx > TP) & (adx_prev <= TP)
    
    # Crossunder: Current < SL and Previous >= SL
    crossunder = (adx < SL) & (adx_prev >= SL)
    
    # SMA Filter Condition
    if SMA_FILTER:
        sma_condition = (sma_short > sma_long)
    else:
        sma_condition = np.ones_like(adx, dtype=bool)
    
    # Entry requires Crossover and SMA Filter
    entry = crossover & sma_condition
    
    # Exit requires Crossunder
    exit_sig = crossunder
    
    # Generate Signals with State Machine
    # Signals are shifted by 1 bar to mimic Pine next-bar execution
    signals = np.zeros(len(df), dtype=float)
    position = 0.0
    
    for i in range(len(df) - 1):
        if position == 0.0:
            if entry[i]:
                position = 1.0
        elif position == 1.0:
            if exit_sig[i]:
                position = 0.0
        signals[i+1] = position
        
    return signals