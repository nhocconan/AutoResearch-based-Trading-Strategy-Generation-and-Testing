#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) with 1d EMA34 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) and price > 1d EMA34, short when > -20 (overbought) and price < 1d EMA34.
# Volume spike (>2.0x 24-bar average) confirms momentum. Discrete sizing 0.25.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via mean reversion from oversold, in bear via shorting overbought rallies.
# Williams %R is a momentum oscillator that identifies overbought/oversold levels, effective in ranging and trending markets when combined with trend filter.

name = "6h_WilliamsR_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14) - using prior completed bar to avoid look-ahead
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = williams_r.values  # Convert to numpy array
    
    # Calculate 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 24-bar average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient warmup
        # Get current values
        wr = williams_r[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(wr) or np.isnan(ema_trend):
            continue
            
        # Entry conditions
        # Long: Williams %R < -80 (oversold) with volume spike and above 1d EMA34
        long_entry = (wr < -80) and (close[i] > ema_trend) and vol_spike
        # Short: Williams %R > -20 (overbought) with volume spike and below 1d EMA34
        short_entry = (wr > -20) and (close[i] < ema_trend) and vol_spike
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Williams %R > -50 (recovering from oversold) or reverse signal
            if wr > -50 or short_entry:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Williams %R < -50 (declining from overbought) or reverse signal
            if wr < -50 or long_entry:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals