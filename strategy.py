#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d EMA trend filter and volume spike confirmation.
# Long: Williams %R < -80 (oversold) + price > 1d EMA34 (uptrend) + volume > 2.0x 20-period average volume.
# Short: Williams %R > -20 (overbought) + price < 1d EMA34 (downtrend) + volume > 2.0x 20-period average volume.
# Exit: Williams %R crosses back through -50 (mean reversion midpoint).
# Uses discrete sizing 0.25 to manage drawdown in volatile markets.
# Williams %R identifies exhaustion points; 1d EMA34 filters for higher timeframe trend alignment;
# volume spike confirms institutional participation at turning points.
# Works in bull markets via buying oversold dips in uptrends and in bear markets via selling overbought rallies in downtrends.

name = "6h_WilliamsR_MeanReversion_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) - measures overbought/oversold levels
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe (wait for 1d bar to close)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # LONG: Oversold + Uptrend + Volume Spike
        if (williams_r[i] < -80 and 
            close[i] > ema_34_aligned[i] and 
            volume[i] > 2.0 * avg_volume[i]):
            signals[i] = 0.25
        # SHORT: Overbought + Downtrend + Volume Spike
        elif (williams_r[i] > -20 and 
              close[i] < ema_34_aligned[i] and 
              volume[i] > 2.0 * avg_volume[i]):
            signals[i] = -0.25
        # EXIT: Mean reversion - Williams %R crosses back through -50
        elif williams_r[i] > -50 and i > 0 and williams_r[i-1] <= -50:
            signals[i] = 0.0  # Exit long
        elif williams_r[i] < -50 and i > 0 and williams_r[i-1] >= -50:
            signals[i] = 0.0  # Exit short
        else:
            # Hold previous signal
            signals[i] = signals[i-1]
    
    return signals