#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Choppiness Index regime filter with 1-week EMA34 trend and volume confirmation.
# Long when: CHOP > 61.8 (ranging market) + price near lower Bollinger Band + EMA34 rising + volume spike.
# Short when: CHOP > 61.8 + price near upper Bollinger Band + EMA34 falling + volume spike.
# Exit when: CHOP < 38.2 (trending market) or opposite signal.
# Designed for ~10-25 trades/year per symbol to avoid fee drag in low-volatility regimes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Choppiness Index (14-period)
    atr14 = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean()
    sum_atr14 = atr14.rolling(window=14, min_periods=14).sum()
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(sum_atr14 / (highest_high14 - lowest_low14)) / np.log10(14)
    chop = chop.values
    
    # Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    lower_bb = sma20 - 2 * std20
    upper_bb = sma20 + 2 * std20
    lower_bb = lower_bb.values
    upper_bb = upper_bb.values
    
    # 1-week EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_filter = volume > (vol_ma20 * 2.0)
    vol_ma20 = vol_ma20.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop[i]) or np.isnan(lower_bb[i]) or np.isnan(upper_bb[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: ranging market (CHOP > 61.8), price near lower BB, EMA34 rising, volume spike
        if (chop[i] > 61.8 and 
            close[i] <= lower_bb[i] * 1.02 and  # within 2% of lower BB
            ema34_1w_aligned[i] > ema34_1w_aligned[i-1] and  # EMA rising
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: ranging market, price near upper BB, EMA34 falling, volume spike
        elif (chop[i] > 61.8 and 
              close[i] >= upper_bb[i] * 0.98 and  # within 2% of upper BB
              ema34_1w_aligned[i] < ema34_1w_aligned[i-1] and  # EMA falling
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: trending market (CHOP < 38.2) or opposite signal
        elif chop[i] < 38.2:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_ChoppinessIndex_1wEMA34_VolumeFilter_BB"
timeframe = "1d"
leverage = 1.0