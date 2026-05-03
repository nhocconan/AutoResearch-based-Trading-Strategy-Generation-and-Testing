#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1w EMA200 trend filter and volume confirmation
# Williams %R measures overbought/oversold: %R = (Highest High - Close)/(Highest High - Lowest Low) * -100
# In bull markets: buy when %R crosses above -80 from below + price above 1w EMA200 + volume spike
# In bear markets: sell when %R crosses below -20 from above + price below 1w EMA200 + volume spike
# Works in both regimes by capturing momentum reversals at extremes
# Volume spike (>2.0x 24-period EMA) confirms institutional participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

name = "12h_WilliamsR_1wEMA200_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(200) for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Williams %R(14) on 12h data
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50.0  # neutral when range is zero
    )
    
    # Volume confirmation: 24-period EMA on 12h volume (2x lookback for 12h)
    vol_series = pd.Series(volume)
    vol_ema_24 = vol_series.ewm(span=24, adjust=False, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(williams_r[i-1]) or np.isnan(vol_ema_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 24-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_24[i])
        
        # Williams %R signals with 1w trend filter
        # Long: %R crosses above -80 from below + price above 1w EMA200 + volume spike
        # Short: %R crosses below -20 from above + price below 1w EMA200 + volume spike
        if position == 0:
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and  # Cross above -80
                close[i] > ema_200_1w_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and  # Cross below -20
                  close[i] < ema_200_1w_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: %R rises above -20 (overbought) OR price below 1w EMA200
            if williams_r[i] > -20 or close[i] < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: %R falls below -80 (oversold) OR price above 1w EMA200
            if williams_r[i] < -80 or close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals