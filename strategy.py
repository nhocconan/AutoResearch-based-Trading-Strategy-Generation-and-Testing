#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian channel breakout with 4h volume spike and chop regime filter
# Long when price breaks above 12h Donchian(20) upper band AND 4h volume > 2.0 * 20-period avg volume AND chop > 61.8 (range)
# Short when price breaks below 12h Donchian(20) lower band AND 4h volume > 2.0 * 20-period avg volume AND chop > 61.8 (range)
# Exit when price crosses 12h Donchian(20) midpoint OR chop < 38.2 (trending)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 12h Donchian provides structure from higher timeframe to avoid noise
# Volume spike confirms breakout strength and reduces false signals
# Chop regime filter ensures we only trade in ranging markets where mean reversion works
# Works in bull markets (buying range highs in uptrend) and bear markets (selling range lows in downtrend)

name = "4h_Donchian20_12h_VolumeSpike_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian channel
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need at least 20 completed 12h bars for Donchian
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian(20) channels
    highest_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high_20
    donchian_lower = lowest_low_20
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align 12h Donchian levels to 4h timeframe (wait for completed 12h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Calculate 4h volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Calculate chop regime filter on 4h: Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending
    # Chop = 100 * log10(sum(ATR(1), n) / (log10(n) * (max(high,n) - min(low,n))))
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr2[0] = high[0] - low[0]  # First period TR
    atr1 = pd.Series(tr2).rolling(window=1, min_periods=1).sum().values
    sum_atr = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (np.log10(14) * (highest_high - lowest_low)))
    chop_regime = chop > 61.8  # Only trade in ranging markets
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(avg_volume_20[i]) or 
            np.isnan(chop[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 12h Donchian upper band, volume confirmation, chop regime (ranging), in session
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                volume_confirm[i] and chop_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 12h Donchian lower band, volume confirmation, chop regime (ranging), in session
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  volume_confirm[i] and chop_regime[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below 12h Donchian midpoint OR chop < 38.2 (trending regime)
            if close[i] < donchian_mid_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above 12h Donchian midpoint OR chop < 38.2 (trending regime)
            if close[i] > donchian_mid_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals