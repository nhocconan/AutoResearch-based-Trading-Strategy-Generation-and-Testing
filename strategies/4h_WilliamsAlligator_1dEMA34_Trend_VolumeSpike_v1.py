#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator (jaw=13, teeth=8, lips=5) identifies trend presence via aligned SMAs
# Long when lips > teeth > jaw + price > 1d EMA(34) + volume spike
# Short when lips < teeth < jaw + price < 1d EMA(34) + volume spike
# Uses discrete position sizing (0.25) to minimize fee churn
# Targets 20-40 trades/year (80-160 total over 4 years) to stay within fee drag limits for 4h timeframe

name = "4h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Williams Alligator SMAs on 4h timeframe
    # Jaw: SMA(13) of median price, Teeth: SMA(8), Lips: SMA(5)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator SMAs and volume MA)
    start_idx = 50  # buffer for 20-period volume MA and 13-period jaw
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: lips > teeth > jaw (bullish alignment) + price > 1d EMA + volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: lips < teeth < jaw (bearish alignment) + price < 1d EMA + volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator alignment breaks (lips <= teeth or teeth <= jaw) or price < 1d EMA
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks (lips >= teeth or teeth >= jaw) or price > 1d EMA
            if lips[i] >= teeth[i] or teeth[i] >= jaw[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals