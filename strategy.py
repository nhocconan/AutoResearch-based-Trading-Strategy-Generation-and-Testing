# 1D_KAMA_Trend_with_Volume_and_ChopFilter
# Hypothesis: Uses daily KAMA direction as primary trend filter, combined with volume confirmation and Choppiness Index regime filter.
# Enters long when KAMA is rising, price > KAMA, volume spike, and market is trending (CHOP < 38.2).
# Enters short when KAMA is falling, price < KAMA, volume spike, and market is trending (CHOP < 38.2).
# Exits when KAMA direction changes or volume drops.
# Uses 1h volume for confirmation (more responsive than daily volume).
# Position size: 0.25 to balance risk and return.
# Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag.
# Works in both bull and bear markets by adapting to trending regimes only.
# Focus on BTC and ETH as primary targets.
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1D_KAMA_Trend_with_Volume_and_ChopFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for volume confirmation (more responsive)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    # Get weekly data for regime filter (Choppiness Index)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate KAMA (2-period ER, 30-period SMA for smoothing)
    # ER = |close - close[10]| / sum(|close - close[-1]| over 10 periods)
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = np.abs(close[i] - close[i-1])
    
    # Sum of volatility over 10 periods
    vol_sum = np.zeros_like(close)
    vol_sum[10:] = np.convolve(volatility, np.ones(10), mode='valid')
    
    # Avoid division by zero
    er = np.zeros_like(close)
    er[10:] = change[10:] / np.where(vol_sum[10:] == 0, 1, vol_sum[10:])
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1h volume average for spike detection
    vol_1h = df_1h['volume'].values
    vol_ma_1h = pd.Series(vol_1h).rolling(window=24, min_periods=24).mean().values  # 24h = 1 day
    vol_ma_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_ma_1h)
    
    # Calculate Choppiness Index on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], 
                     np.maximum(np.abs(high_1w[1:] - close_1w[:-1]), 
                                np.abs(low_1w[1:] - close_1w[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(tr1) / (hh - ll)) / log10(14)
    # Avoid division by zero and log of zero
    range_hl = hh - ll
    chop = np.full_like(close_1w, 50.0)  # default to neutral
    mask = (range_hl > 0) & (~np.isnan(tr_sum))
    chop[mask] = 100 * np.log10(tr_sum[mask] / range_hl[mask]) / np.log10(14)
    
    # Align Chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Volume spike: current 1h volume > 1.5 * 24-period average
    vol_spike = volume > (1.5 * vol_ma_1h_aligned)
    
    # KAMA direction: rising if current > previous, falling if current < previous
    kama_dir = np.diff(kama, prepend=kama[0])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for KAMA and other indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_1h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama[i]
        chop_val = chop_aligned[i]
        vol_spike_bool = vol_spike[i]
        kama_direction = kama_dir[i]
        
        if position == 0:
            # Enter long: KAMA rising, price > KAMA, volume spike, trending market (CHOP < 38.2)
            if (kama_direction > 0 and price > kama_val and vol_spike_bool and chop_val < 38.2):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, price < KAMA, volume spike, trending market (CHOP < 38.2)
            elif (kama_direction < 0 and price < kama_val and vol_spike_bool and chop_val < 38.2):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA falling OR no volume spike OR chop becomes too high (range)
            if (kama_direction <= 0 or not vol_spike_bool or chop_val > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA rising OR no volume spike OR chop becomes too high (range)
            if (kama_direction >= 0 or not vol_spike_bool or chop_val > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals