#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout for trend direction and 1d volume confirmation for entry timing
# - Uses 4h Donchian channel (20-period) to identify trend direction
# - Uses 1d volume spike (2x 20-day average) to confirm institutional participation
# - Enters long when price breaks above 4h upper Donchian band with volume spike in uptrend
# - Enters short when price breaks below 4h lower Donchian band with volume spike in downtrend
# - Exits when price returns to 4h Donchian middle band (mean reversion within the channel)
# - Uses session filter (08-20 UTC) to avoid low-liquidity periods
# - Position size: 0.20 (20% of capital) to manage risk during drawdowns
# - Designed to capture breakouts with institutional validation while minimizing false signals
# - Target: 60-150 total trades over 4 years (15-37/year) with controlled frequency

name = "1h_4hDonchian_1dVolume_Breakout"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian Channel (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band: highest high over 20 periods
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 periods
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate 1d volume spike (2x 20-day average)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_20)
    
    # Align 4h indicators to 1h timeframe
    donchian_upper_1h = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_1h = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_1h = align_htf_to_ltf(prices, df_4h, donchian_middle)
    
    # Align 1d volume spike to 1h timeframe
    volume_spike_1h = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Session filter: 08-20 UTC (avoid low-liquidity periods)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_upper_1h[i]) or np.isnan(donchian_lower_1h[i]) or 
            np.isnan(donchian_middle_1h[i]) or np.isnan(volume_spike_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h upper Donchian band with volume spike
            if close[i] > donchian_upper_1h[i] and volume_spike_1h[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h lower Donchian band with volume spike
            elif close[i] < donchian_lower_1h[i] and volume_spike_1h[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to 4h Donchian middle band (mean reversion)
            if close[i] <= donchian_middle_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to 4h Donchian middle band (mean reversion)
            if close[i] >= donchian_middle_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals