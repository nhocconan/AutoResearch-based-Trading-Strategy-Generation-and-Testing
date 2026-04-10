#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d trend filter + volume confirmation
# - Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend absence when lines are intertwined
# - Enter long when price > Alligator Lips AND 1d close > 1d EMA50 AND volume > 1.3x 20-period average
# - Enter short when price < Alligator Lips AND 1d close < 1d EMA50 AND volume > 1.3x 20-period average
# - Exit when price crosses back below/above Alligator Teeth (8-period smoothed median)
# - Uses 12h primary timeframe with 1d HTF for trend alignment
# - Targets 12-25 trades/year (50-100 total over 4 years) to avoid fee drag
# - Alligator excels in ranging markets by identifying trend absence; EMA filter ensures direction alignment with higher timeframe

name = "12h_1d_alligator_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 12h prices (Jaw=13, Teeth=8, Lips=5)
    # Alligator uses smoothed median (SMMA) - approximated with EMA for simplicity
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Lips (5-period SMMA of median price, shifted 3 bars)
    median_price = (high + low) / 2.0
    lips_raw = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # shift 3 bars forward
    lips[:3] = lips_raw[3] if not np.isnan(lips_raw[3]) else 0  # fill first 3 values
    
    # Teeth (8-period SMMA of median price, shifted 5 bars)
    teeth_raw = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # shift 5 bars forward
    teeth[:5] = teeth_raw[5] if not np.isnan(teeth_raw[5]) else 0  # fill first 5 values
    
    # Jaw (13-period SMMA of median price, shifted 8 bars)
    jaw_raw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # shift 8 bars forward
    jaw[:8] = jaw_raw[8] if not np.isnan(jaw_raw[8]) else 0  # fill first 8 values
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Lips AND 1d uptrend AND volume spike
            if (prices['close'].iloc[i] > lips[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i] and
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price < Lips AND 1d downtrend AND volume spike
            elif (prices['close'].iloc[i] < lips[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i] and
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when price crosses back below/above Teeth (8-period)
            if position == 1:  # Long position
                if prices['close'].iloc[i] < teeth[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if prices['close'].iloc[i] > teeth[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals