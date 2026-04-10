#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout + volume confirmation + session filter
# - Long when price breaks above 4h Donchian upper channel (20-period) AND volume > 2.0x 20-period average AND hour in 08-20 UTC
# - Short when price breaks below 4h Donchian lower channel (20-period) AND volume > 2.0x 20-period average AND hour in 08-20 UTC
# - Exit when price returns to 4h Donchian middle (mean reversion)
# - Uses discrete position sizing 0.20 to limit fee churn
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - 4h Donchian provides institutional support/resistance that work in both trending and ranging markets
# - Volume confirmation reduces false breakouts
# - Session filter avoids low-liquidity periods

name = "1h_4h_donchian_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Pre-compute 1h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute 1h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper channel: highest high over last 20 periods
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower channel: lowest low over last 20 periods
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Donchian middle: average of upper and lower
    donchian_middle = (donchian_high + donchian_low) / 2
    
    # Align HTF indicators to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(vol_ma[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian upper channel AND volume spike AND session
            if (close[i] > donchian_high_aligned[i] and 
                volume_spike[i] and 
                session_filter[i]):
                position = 1
                signals[i] = 0.20
            # Short conditions: price breaks below Donchian lower channel AND volume spike AND session
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_spike[i] and 
                  session_filter[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to middle (mean reversion)
            # Exit when price returns to Donchian middle (mean reversion to equilibrium)
            exit_long = (position == 1 and close[i] <= donchian_middle_aligned[i])
            exit_short = (position == -1 and close[i] >= donchian_middle_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals