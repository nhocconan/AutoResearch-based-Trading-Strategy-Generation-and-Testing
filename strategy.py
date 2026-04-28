#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily Williams %R extremes with 1d EMA34 trend filter and volume spike confirmation.
# Enter long when Williams %R < -80 (oversold) with volume spike and price above 1d EMA34.
# Enter short when Williams %R > -20 (overbought) with volume spike and price below 1d EMA34.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.
# Williams %R is a proven momentum oscillator that works well in ranging markets and catch reversals in trends.
# The 1d EMA34 filter ensures we trade with the higher timeframe trend, reducing whipsaws.
# Volume spike confirms institutional participation at turning points.

name = "6h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams %R and EMA
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams %R on 1d data: (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n_1d = len(high_1d)
    williams_r = np.full(n_1d, np.nan)
    
    # Williams %R lookback period
    lookback = 14
    
    for i in range(lookback - 1, n_1d):
        highest_high = np.max(high_1d[i - lookback + 1:i + 1])
        lowest_low = np.min(low_1d[i - lookback + 1:i + 1])
        if highest_high != lowest_low:
            williams_r[i] = ((highest_high - close_1d[i]) / (highest_high - lowest_low)) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Forward fill Williams %R
    williams_r = pd.Series(williams_r).ffill().values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe with 1-bar delay for confirmation
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R extreme levels
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # Trend filter: price relative to 1d EMA34
        above_ema = close[i] > ema_34_1d_aligned[i]
        below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = oversold and volume_spike[i] and above_ema
        short_entry = overbought and volume_spike[i] and below_ema
        
        # Exit conditions: opposite extreme or trend reversal
        long_exit = williams_r_aligned[i] > -20 or below_ema  # overbought or trend change
        short_exit = williams_r_aligned[i] < -80 or above_ema  # oversold or trend change
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
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