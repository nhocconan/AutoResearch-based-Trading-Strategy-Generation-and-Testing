#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d RSI Mean Reversion + Volume Spike
# Williams Alligator identifies trend direction via three smoothed moving averages (Jaw, Teeth, Lips).
# Long when Lips > Teeth > Jaw (bullish alignment) and RSI < 40 (oversold).
# Short when Lips < Teeth < Jaw (bearish alignment) and RSI > 60 (overbought).
# Volume confirmation requires > 1.5x 20-bar median volume.
# Designed to work in bull markets (trend following via Alligator) and bear markets (mean reversion via RSI extremes).
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day RSI(14) for mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # Williams Alligator on 4h close
    # Jaw: 13-period SMMA, 8-period offset
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # 8-period offset
    # Teeth: 8-period SMMA, 5-period offset
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # 5-period offset
    # Lips: 5-period SMMA, 3-period offset
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # 3-period offset
    
    jaw_arr = jaw.values
    teeth_arr = teeth.values
    lips_arr = lips.values
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(lips_arr[i]) or np.isnan(teeth_arr[i]) or np.isnan(jaw_arr[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = lips_arr[i] > teeth_arr[i] and teeth_arr[i] > jaw_arr[i]
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_alignment = lips_arr[i] < teeth_arr[i] and teeth_arr[i] < jaw_arr[i]
        
        # Long: Bullish alignment, RSI oversold (<40), volume spike
        if (bullish_alignment and 
            rsi_1d_aligned[i] < 40 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Bearish alignment, RSI overbought (>60), volume spike
        elif (bearish_alignment and 
              rsi_1d_aligned[i] > 60 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Alligator alignment breaks or RSI returns to neutral (40-60)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (not bullish_alignment or rsi_1d_aligned[i] >= 40)) or
               (signals[i-1] == -0.25 and (not bearish_alignment or rsi_1d_aligned[i] <= 60)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_WilliamsAlligator_RSI1d_Volume"
timeframe = "4h"
leverage = 1.0