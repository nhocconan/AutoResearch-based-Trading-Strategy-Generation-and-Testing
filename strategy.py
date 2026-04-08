# USING 12H CAMARILLA + 6H MOMENTUM
# - 12h Camarilla levels from previous day (R3/S3, R4/S4)
# - 6h momentum: price > SMA20 and RSI > 50 for long, < for short
# - Volume filter: volume > 1.5x 20-period average
# - Works in bull/bear: Camarilla provides structure, momentum filters direction
# Target: 50-150 total trades over 4 years

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_momentum_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for Camarilla levels (from previous day)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    # Using previous day's range (for 12h, we use previous 12h bar's range)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # But standard Camarilla uses daily range
    # For 12h timeframe, we'll use the 12h bar's range as proxy for "daily"
    range_12h = high_12h - low_12h
    camarilla_r4 = close_12h + range_12h * 1.1 / 2
    camarilla_r3 = close_12h + range_12h * 1.1 / 4
    camarilla_s3 = close_12h - range_12h * 1.1 / 4
    camarilla_s4 = close_12h - range_12h * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_6h = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_r3_6h = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_s4_6h = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # 6h momentum indicators
    # SMA20 on 6h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    # RSI(14) on 6h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14 = rsi_14.values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_20[i]) or np.isnan(rsi_14[i]) or
            np.isnan(camarilla_r4_6h[i]) or np.isnan(camarilla_r3_6h[i]) or
            np.isnan(camarilla_s3_6h[i]) or np.isnan(camarilla_s4_6h[i]) or
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < S3 or momentum fails
            if close[i] < camarilla_s3_6h[i] or rsi_14[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > R3 or momentum fails
            if close[i] > camarilla_r3_6h[i] or rsi_14[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Momentum filter
            bullish = close[i] > sma_20[i] and rsi_14[i] > 50
            bearish = close[i] < sma_20[i] and rsi_14[i] < 50
            
            # Long: price > R4 + bullish momentum + volume
            if (close[i] > camarilla_r4_6h[i] and 
                bullish and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price < S4 + bearish momentum + volume
            elif (close[i] < camarilla_s4_6h[i] and 
                  bearish and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals