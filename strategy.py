#!/usr/bin/env python3
"""
exp_6493_4h_donchian20_12h_hma_vol_v1
Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation.
Uses 12h HMA(21) as trend filter for smoother trend detection than EMA, reducing whipsaw.
Donchian breakout provides entry timing, volume confirmation filters weak breakouts.
Designed to work in both bull and bear markets by using 12h HMA as trend filter.
Target: 75-200 trades over 4 years (19-50/year). Uses 4h primary timeframe per experiment instructions.
"""
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6493_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_PERIOD = 21
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 1.8  # volume must be 1.8x its 20-period MA
SIGNAL_SIZE = 0.25   # 25% position size

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean()
    # WMA of full period
    wma_full = pd.Series(series).ewm(span=period, adjust=False).mean()
    # Raw HMA
    raw_hma = 2 * wma_half - wma_full
    # Final HMA
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
    return hma.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for HMA trend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA(21)
    close_12h = df_12h['close'].values
    hma_12h = calculate_hma(close_12h, HMA_PERIOD)
    
    # Align to LTF (4h) with shift(1) for completed bars only
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, HMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HMA data not available
        if np.isnan(hma_12h_aligned[i]):
            continue
            
        # Long conditions: price breaks above Donchian HIGH + above 12h HMA + volume spike
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_trend = close[i] > hma_12h_aligned[i]  # price above 12h HMA (bullish trend)
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price breaks below Donchian LOW + below 12h HMA + volume spike
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_trend = close[i] < hma_12h_aligned[i]  # price below 12h HMA (bearish trend)
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: simple midpoint reversal
        if position == 1:  # long position
            # Exit if price drops below midpoint of channel
            exit_long = close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Or if price breaks below Donchian low (strong reversal)
            exit_long = exit_long or close[i] < donchian_low[i-1]
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises above midpoint of channel
            exit_short = close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Or if price breaks above Donchian high (strong reversal)
            exit_short = exit_short or close[i] > donchian_high[i-1]
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout and long_trend and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_breakout and short_trend and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals