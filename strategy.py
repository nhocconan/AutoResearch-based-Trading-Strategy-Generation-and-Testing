#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining Donchian breakout with 1d volatility filter and volume confirmation.
# Long when price breaks above Donchian(20) high with above-average volatility and volume.
# Short when price breaks below Donchian(20) low with above-average volatility and volume.
# Exit when price returns to Donchian middle or volatility drops below average.
# Uses volatility filter to avoid whipsaws in low-volatility periods.
# Designed for 12h timeframe to reduce trade frequency and minimize fee drag.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Donchian channels and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate ATR for volatility filter
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_period = 14
    atr_1d = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate average ATR for volatility regime filter
    atr_ma = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    
    # Align indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid)
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need Donchian and volatility MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(donch_mid_aligned[i]) or
            np.isnan(atr_ma_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is above average
        high_volatility = atr_1d[i] > atr_ma_aligned[i] if not np.isnan(atr_1d[i]) else False
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Look for Donchian breakouts with volatility and volume confirmation
            # Long: price breaks above Donchian high
            if (close[i] > donch_high_aligned[i] and 
                high_volatility and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low
            elif (close[i] < donch_low_aligned[i] and 
                  high_volatility and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian middle or volatility drops
            if (close[i] <= donch_mid_aligned[i] or 
                not high_volatility):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian middle or volatility drops
            if (close[i] >= donch_mid_aligned[i] or 
                not high_volatility):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_DonchianBreakout_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0