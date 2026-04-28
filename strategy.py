#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R (14) extreme reversals with volume confirmation and 1d EMA200 trend filter.
# Enter long when Williams %R crosses above -80 from oversold with volume spike and price above 1d EMA200.
# Enter short when Williams %R crosses below -20 from overbought with volume spike and price below 1d EMA200.
# Uses discrete position sizing (0.25) to balance return and drawdown. Target: 12-37 trades/year.
# Williams %R provides timely reversal signals, volume confirms momentum, EMA200 filters intermediate trend.
# Works in bull (buying dips in uptrend) and bear (selling rallies in downtrend) markets.

name = "12h_WilliamsR14_1dEMA200_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA200 (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n_1d = len(high_1d)
    williams_r = np.full(n_1d, np.nan)
    
    for i in range(14, n_1d):
        highest_high = np.max(high_1d[i-14:i+1])
        lowest_low = np.min(low_1d[i-14:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_1d[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 12h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA200
        above_ema = close[i] > ema_200_1d_aligned[i]
        below_ema = close[i] < ema_200_1d_aligned[i]
        
        # Williams %R reversal conditions with volume confirmation
        # Long: crosses above -80 from oversold
        long_reversal = (williams_r_aligned[i] > -80 and 
                         williams_r_aligned[i-1] <= -80 and 
                         volume_spike[i])
        # Short: crosses below -20 from overbought
        short_reversal = (williams_r_aligned[i] < -20 and 
                          williams_r_aligned[i-1] >= -20 and 
                          volume_spike[i])
        
        # Exit conditions: opposite extreme or trend reversal
        long_exit = (williams_r_aligned[i] < -20) or below_ema
        short_exit = (williams_r_aligned[i] > -80) or above_ema
        
        # Handle entries and exits
        if long_reversal and above_ema and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_reversal and below_ema and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals