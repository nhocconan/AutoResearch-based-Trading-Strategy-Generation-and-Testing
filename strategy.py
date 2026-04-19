#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator with daily ATR filter and volume spike confirmation.
# Long when: Jaw < Teeth < Lips (bullish alignment), price > Lips, volume > 2x 20-period average
# Short when: Jaw > Teeth > Lips (bearish alignment), price < Lips, volume > 2x 20-period average
# Exit when: Alignment breaks (Teeth crosses Jaw or Lips)
# Williams Alligator identifies trend phases, daily ATR filter avoids low-volatility chop, volume confirms strength.
# Target: 15-25 trades/year per symbol. Works in bull (ride trends) and bear (catch reversals).
name = "12h_WilliamsAlligator_ATR_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Williams Alligator and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMAs of median price (H+L)/2
    median_price = (high_1d + low_1d) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # Red line
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # Green line
    
    # ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1D data to 12H timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volatility filter: avoid low-volatility chop
        vol_filter = atr_val > (np.nanmedian(atr_aligned[max(0, i-50):i+1]) * 0.8)
        
        if position == 0:
            # Bullish alignment: Jaw < Teeth < Lips
            bullish = jaw_val < teeth_val < lips_val
            # Bearish alignment: Jaw > Teeth > Lips
            bearish = jaw_val > teeth_val > lips_val
            
            # Long entry: Bullish alignment, price > Lips, volume spike, sufficient volatility
            if (bullish and price > lips_val and vol > 2.0 * vol_ma and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish alignment, price < Lips, volume spike, sufficient volatility
            elif (bearish and price < lips_val and vol > 2.0 * vol_ma and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bullish alignment breaks (Teeth crosses below Jaw OR Lips crosses below Teeth)
            if not (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bearish alignment breaks (Teeth crosses above Jaw OR Lips crosses above Teeth)
            if not (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals