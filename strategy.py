#!/usr/bin/env python3
"""
Experiment #8175: 6-hour ADX trend with 1-day RSI mean reversion and volume confirmation.
Hypothesis: In trending markets (ADX > 25), pullbacks to RSI extremes (RSI < 30 for long, RSI > 70 for short) 
on the 6h timeframe offer high-probability entries with the trend. Volume confirmation ensures 
institutional participation. This combination works in both bull and bear markets by only trading 
in the direction of the higher timeframe trend (1d ADX), avoiding counter-trend whipsaws.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8175_6h_adx_rsi_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_rsi(prices, period=14):
    """Calculate RSI using Wilder's smoothing method."""
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (same as alpha = 1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth the values
    tr_sum = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_sum = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_sum = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * np.where(tr_sum != 0, dm_plus_sum / tr_sum, 0)
    di_minus = 100 * np.where(tr_sum != 0, dm_minus_sum / tr_sum, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - 1d for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, ADX_PERIOD)
    
    # Strong trend filter: ADX > 25
    strong_trend = adx_1d > ADX_THRESHOLD
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI for mean reversion entries
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Volume moving average for confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, RSI_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(strong_trend_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market conditions
        is_strong_trend = strong_trend_aligned[i]
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # RSI conditions for mean reversion entries
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        
        # Entry conditions - only trade pullbacks in strong trends
        long_entry = is_strong_trend and rsi_oversold and volume_confirmed
        short_entry = is_strong_trend and rsi_overbought and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals