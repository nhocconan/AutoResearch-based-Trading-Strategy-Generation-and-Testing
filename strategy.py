# Implementing strategy: 1h_4hDonchian_1dVolumeSpike_SessionFilter
# Hypothesis: Use 4h Donchian channel breakout for trend direction, confirmed by 1d volume spike (institutional interest).
# Enter on 1h pullback to VWAP during London/NY session (08-20 UTC). Avoids chop and false breakouts.
# Designed for 1h timeframe with selective entries to stay within 15-37 trades/year.
# Works in bull (breakouts continue) and bear (failed breaks reverse to VWAP).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for Donchian channel (trend direction)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian(20) on 4h
    highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_high = align_htf_to_ltf(prices, df_4h, highest_high_20)
    donchian_low = align_htf_to_ltf(prices, df_4h, lowest_low_20)
    
    # Load 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day average volume on 1d
    avg_vol_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * avg_vol_20)  # 50% above average = institutional interest
    
    # Align 1d volume spike to 1h timeframe
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate 1h VWAP for entry timing
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    vwap = (typical_price * prices['volume']).cumsum() / prices['volume'].cumsum()
    vwap = vwap.values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if NaN in any indicator
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vwap[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # London/NY session
        
        if position == 0:
            # Long: price breaks above 4h Donchian high + volume spike + pullback to VWAP in session
            if (price > donchian_high[i] and 
                vol_spike_aligned[i] > 0.5 and 
                abs(price - vwap[i]) / vwap[i] < 0.005 and  # within 0.5% of VWAP
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low + volume spike + pullback to VWAP in session
            elif (price < donchian_low[i] and 
                  vol_spike_aligned[i] > 0.5 and 
                  abs(price - vwap[i]) / vwap[i] < 0.005 and
                  in_session):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 4h Donchian low or outside session
            if price < donchian_low[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above 4h Donchian high or outside session
            if price > donchian_high[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hDonchian_1dVolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0