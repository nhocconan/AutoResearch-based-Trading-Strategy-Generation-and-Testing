#!/usr/bin/env python3
"""
1d_TRIX_VolumeSpike_Regime
Hypothesis: Uses TRIX momentum crossover combined with volume spikes and Choppiness Index regime filter.
TRIX captures momentum changes; volume spikes confirm institutional interest; Choppiness Index filters for trending vs ranging markets.
Works in bull markets by catching strong momentum and in bear markets by identifying mean-reversion opportunities in ranging conditions.
Target: 15-25 trades/year per symbol (30-100 total over 4 years).
"""

name = "1d_TRIX_VolumeSpike_Regime"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Convert to Series for indicator calculations
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # TRIX (15-period)
    # Single EMA
    ema1 = close_s.ewm(span=15, adjust=False, min_periods=15).mean()
    # Double EMA
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    # Triple EMA
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    # TRIX = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix.fillna(0).values
    trix_prev = np.roll(trix, 1)
    trix_prev[0] = 0
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period)
    # True Range
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).mean()
    
    # Sum of True Range over 14 periods
    sum_tr14 = tr.rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh14 = high_s.rolling(window=14, min_periods=14).max()
    ll14 = low_s.rolling(window=14, min_periods=14).min()
    
    # Chop = 100 * log10(sum_tr14 / (hh14 - ll14)) / log10(14)
    # Avoid division by zero
    range_hl = hh14 - ll14
    chop = 100 * np.log10(sum_tr14 / range_hl.replace(0, np.nan)) / np.log10(14)
    chop = chop.replace([np.inf, -np.inf], np.nan).fillna(50).values  # neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(trix_prev[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0
        
        # TRIX crossover signals
        trix_bullish = trix[i] > 0 and trix_prev[i] <= 0
        trix_bearish = trix[i] < 0 and trix_prev[i] >= 0
        
        # Chop regime: > 61.8 = ranging (mean revert), < 38.2 = trending (trend follow)
        chop_high = chop[i] > 61.8  # ranging market
        chop_low = chop[i] < 38.2   # trending market
        
        if position == 0:
            # Enter long: TRIX bullish cross + volume + trending OR ranging with mean reversion bias
            if (trix_bullish and volume_confirm and
                (chop_low or (chop_high and trix[i] < -0.1))):  # in ranging, look for oversold bounce
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX bearish cross + volume + trending OR ranging with mean reversion bias
            elif (trix_bearish and volume_confirm and
                  (chop_low or (chop_high and trix[i] > 0.1))):  # in ranging, look for overbought fade
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when momentum fades or chop indicates strong ranging
            if (trix[i] < 0 or chop[i] > 70):  # strong ranging or momentum loss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when momentum fades or chop indicates strong ranging
            if (trix[i] > 0 or chop[i] > 70):  # strong ranging or momentum loss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals