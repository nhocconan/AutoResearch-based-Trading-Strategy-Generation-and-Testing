#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1-day trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions (below -80 = oversold, above -20 = overbought)
# Entry: Long when %R crosses above -80 from below + volume spike + price > 1d EMA50
# Entry: Short when %R crosses below -20 from above + volume spike + price < 1d EMA50
# Exit: When %R returns to opposite threshold (-20 for longs, -80 for shorts)
# Uses 1-day EMA for trend alignment to avoid counter-trend trades
# Williams %R is mean-reverting but works best with trend filter in crypto markets
# Target: 15-25 trades/year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Williams %R calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R on daily data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    
    # Calculate 50-period EMA on 1d close for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (20-period on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 6-hour timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below + volume spike + uptrend
            if (williams_r_aligned[i] > -80 and 
                williams_r_aligned[i-1] <= -80 and 
                vol_spike[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above + volume spike + downtrend
            elif (williams_r_aligned[i] < -20 and 
                  williams_r_aligned[i-1] >= -20 and 
                  vol_spike[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to opposite threshold
            if position == 1:
                if williams_r_aligned[i] >= -20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if williams_r_aligned[i] <= -80:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA50_Volume_Spike_Session"
timeframe = "6h"
leverage = 1.0