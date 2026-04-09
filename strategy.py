#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R with 4h EMA filter and volume spike
# Williams %R identifies overbought/oversold conditions. In 6h timeframe:
# - Long when %R crosses above -80 from below AND price > 20 EMA AND volume > 1.5x avg volume
# - Short when %R crosses below -20 from above AND price < 20 EMA AND volume > 1.5x avg volume
# - Exit when %R crosses opposite threshold (-20 for long exit, -80 for short exit)
# - Uses discrete position sizing 0.25 to target ~12-37 trades/year and minimize fee drag
# - Works in bull/bear markets: mean reversion at extremes in ranging, momentum in trending

name = "6h_1d_williamsr_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_1d) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Calculate 4h EMA(20) for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 4h average volume (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Williams %R to 6h timeframe (no extra delay needed for Williams %R)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit long when Williams %R crosses above -20 (overbought)
            if williams_r_aligned[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short when Williams %R crosses below -80 (oversold)
            if williams_r_aligned[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entry signals
            # Long: Williams %R crosses above -80 from below, price > EMA20, volume confirmed
            long_signal = (
                williams_r_aligned[i] > -80 and 
                williams_r_aligned[i-1] <= -80 and
                close[i] > ema_20[i] and
                volume_confirmed
            )
            
            # Short: Williams %R crosses below -20 from above, price < EMA20, volume confirmed
            short_signal = (
                williams_r_aligned[i] < -20 and 
                williams_r_aligned[i-1] >= -20 and
                close[i] < ema_20[i] and
                volume_confirmed
            )
            
            if long_signal:
                position = 1
                signals[i] = 0.25
            elif short_signal:
                position = -1
                signals[i] = -0.25
    
    return signals