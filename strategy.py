#!/usr/bin/env python3
"""
Experiment #8875: 6h Donchian breakout with 1d/1w trend filter and volume confirmation.
Hypothesis: 6h timeframe balances trade frequency and signal quality. Using 1d and 1w EMA filters ensures
multi-timeframe alignment with dominant trends, reducing counter-trend trades. Volume confirmation filters
for institutional participation. Targets 50-150 total trades over 4 years (12-37/year) to minimize fee impact.
Works in both bull and bear markets by aligning with higher timeframe trend direction.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8875_6h_donchian20_1d_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
TREND_FAST = 20   # 1d EMA for short-term trend
TREND_SLOW = 50   # 1w EMA for long-term trend
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d EMA for short-term trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=TREND_FAST, adjust=False, min_periods=TREND_FAST).mean().values
    
    # Calculate 1w EMA for long-term trend
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=TREND_SLOW, adjust=False, min_periods=TREND_SLOW).mean().values
    
    # Price relative to EMAs: above = bullish bias, below = bearish bias
    trend_1d = np.where(close_1d > ema_1d, 1, 
                 np.where(close_1d < ema_1d, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    trend_1w = np.where(close_1w > ema_1w, 1, 
                 np.where(close_1w < ema_1w, -1, 0))
    
    # Align HTF trends to LTF
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Combined trend: both must agree for signal (reduces whipsaw)
    bullish_bias = (trend_1d_aligned == 1) & (trend_1w_aligned == 1)
    bearish_bias = (trend_1d_aligned == -1) & (trend_1w_aligned == -1)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_FAST, TREND_SLOW, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(trend_1d_aligned[i]) or np.isnan(trend_1w_aligned[i]):
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
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions require both timeframes to agree
        long_entry = bullish_bias[i] and long_breakout and volume_confirmed
        short_entry = bearish_bias[i] and short_breakout and volume_confirmed
        
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
</x>