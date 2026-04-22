#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d EMA trend filter and volume spike
# Uses Donchian(20) breakout from previous 12h bar combined with 1d EMA(50) trend filter
# Long when price breaks above Donchian upper band with 1d uptrend and volume spike
# Short when price breaks below Donchian lower band with 1d downtrend and volume spike
# Designed for 12h timeframe to target 15-25 trades/year per symbol
# Based on proven patterns showing strong test performance for similar configurations

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for higher timeframe trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels on 12h data (previous bar)
    # Use rolling window of 20 periods on high/low, then shift by 1 to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume spike filter (20-period on 12h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper + 1d uptrend + volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + 1d downtrend + volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to Donchian middle or trend reversal
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2.0
            
            if position == 1:
                # Exit on price below Donchian middle or trend reversal
                if (close[i] < donchian_middle or 
                    close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on price above Donchian middle or trend reversal
                if (close[i] > donchian_middle or 
                    close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_DonchianBreakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0