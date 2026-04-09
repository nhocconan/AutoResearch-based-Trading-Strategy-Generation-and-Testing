#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA trend filter and volume confirmation
# - Uses 4h Donchian channels (20-period) for breakout signals
# - Uses 12h HMA(21) for trend direction filter (only trade in trend direction)
# - Requires volume > 1.5x 20-period average for confirmation
# - Exits on opposite Donchian channel touch or trend reversal
# - Position size: 0.25 (25% of capital) to balance risk and return
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to avoid fee drag
# - Works in bull markets (breakouts continue trend) and bear markets (breakouts reverse trends)

name = "4h_12h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC) - optional for 4h
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Upper channel: highest high of last 20 periods
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 periods
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h HMA(21) for trend direction
    close_12h = df_12h['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    n_hma = 21
    n_half = n_hma // 2
    n_sqrt = int(np.sqrt(n_hma))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate HMA manually to avoid look-ahead
    hma_12h = np.full_like(close_12h, np.nan, dtype=float)
    for i in range(n_hma, len(close_12h)):
        # WMA of last n_half prices
        half_slice = close_12h[i - n_half + 1:i + 1]
        if len(half_slice) == n_half:
            wma_half = wma(half_slice, n_half)
            # WMA of last n_hma prices
            full_slice = close_12h[i - n_hma + 1:i + 1]
            if len(full_slice) == n_hma:
                wma_full = wma(full_slice, n_hma)
                # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
                raw_hma = 2 * wma_half - wma_full
                if len(raw_hma) >= n_sqrt:
                    # WMA of the raw HMA with sqrt(n) period
                    hma_slice = np.full(n_sqrt, np.nan)
                    hma_slice[-1] = raw_hma
                    # Need to calculate WMA properly for the last n_sqrt values
                    # Simpler approach: use pandas for WMA calculation
                    hma_12h[i] = pd.Series(raw_hma[-n_sqrt:]).rolling(window=n_sqrt, min_periods=n_sqrt).mean().iloc[-1] if len(raw_hma) >= n_sqrt else np.nan
    
    # Simpler HMA calculation using pandas (equivalent)
    # HMA(21) = WMA(2*WMA(10.5) - WMA(21)), round(sqrt(21))=4
    # Since we can't have half periods, use integer approximation
    half_period = 10
    full_period = 21
    sqrt_period = 4
    
    wma_half = pd.Series(close_12h).rolling(window=half_period, min_periods=half_period).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=False
    ).values
    wma_full = pd.Series(close_12h).rolling(window=full_period, min_periods=full_period).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=False
    ).values
    
    raw_hma = 2 * wma_half - wma_full
    hma_12h = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).mean().values
    
    # Align 12h HMA to 4h
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(hma_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h HMA
        # HMA rising = uptrend, HMA falling = downtrend
        if i > 20 and not np.isnan(hma_12h_aligned[i-1]):
            hma_rising = hma_12h_aligned[i] > hma_12h_aligned[i-1]
            hma_falling = hma_12h_aligned[i] < hma_12h_aligned[i-1]
        else:
            hma_rising = False
            hma_falling = False
        
        if position == 1:  # Long position
            # Exit conditions: touch lower channel or trend turns down
            if close[i] <= lower_channel[i] or not hma_rising:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: touch upper channel or trend turns up
            if close[i] >= upper_channel[i] or not hma_falling:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries aligned with 12h trend
            if (close[i] > upper_channel[i] and  # Break above upper channel
                volume_ok[i] and                 # Volume confirmation
                hma_rising):                     # 12h trend up
                position = 1
                signals[i] = 0.25
            elif (close[i] < lower_channel[i] and  # Break below lower channel
                  volume_ok[i] and                 # Volume confirmation
                  hma_falling):                    # 12h trend down
                position = -1
                signals[i] = -0.25
    
    return signals