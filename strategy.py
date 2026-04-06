#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout (trend direction) + 1d volume confirmation + session filter (08-20 UTC)
# Long when price breaks above 4h Donchian high AND volume > 1.5x 1d average AND session active
# Short when price breaks below 4h Donchian low AND volume > 1.5x 1d average AND session active
# Exit when price crosses 4h midline OR volume < average OR outside session
# Uses 4h trend to avoid whipsaw, volume to confirm breakout, session to reduce noise
# Target: 60-150 total trades over 4 years = 15-37/year for 1h

name = "1h_donchian_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = pd.DatetimeIndex(prices['open_time'])
    hours = open_time.hour  # Pre-compute for session filter
    
    # 4h Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Align to 1h timeframe (shifted by 1 for completed bars only)
    donch_high_1h = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_1h = align_htf_to_ltf(prices, df_4h, donch_low)
    donch_mid_1h = align_htf_to_ltf(prices, df_4h, donch_mid)
    
    # 1d volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_1h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after sufficient warmup
        # Skip if required data not available
        if np.isnan(donch_high_1h[i]) or np.isnan(donch_low_1h[i]) or np.isnan(donch_mid_1h[i]) or np.isnan(vol_1h[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Exit conditions
        if position == 1:  # long position
            if (close[i] <= donch_mid_1h[i] or volume[i] < vol_1h[i] or not in_session):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if (close[i] >= donch_mid_1h[i] or volume[i] < vol_1h[i] or not in_session):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Entry conditions: Donchian breakout + volume confirmation + session
            if in_session and volume[i] > 1.5 * vol_1h[i]:
                if close[i] > donch_high_1h[i]:  # Break above Donchian high
                    signals[i] = 0.20
                    position = 1
                elif close[i] < donch_low_1h[i]:  # Break below Donchian low
                    signals[i] = -0.20
                    position = -1
    
    return signals