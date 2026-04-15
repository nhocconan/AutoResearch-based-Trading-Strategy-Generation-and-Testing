#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d RSI Mean Reversion + Volume Spike
# Williams Alligator: three SMAs (jaw=13, teeth=8, lips=5) to identify trends.
# Long when lips > teeth > jaw (bullish alignment) and RSI < 40 (oversold).
# Short when lips < teeth < jaw (bearish alignment) and RSI > 60 (overbought).
# Volume confirmation requires > 1.5x 20-bar median volume.
# Designed to work in bull markets (trend following) and bear markets (mean reversion via RSI extremes).
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
    # Jaw: 13-period SMA, shifted 8 bars forward
    # Teeth: 8-period SMA, shifted 5 bars forward
    # Lips: 5-period SMA, shifted 3 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    
    # Shift to align with Alligator's predictive nature
    jaw_shifted = jaw.shift(8)
    teeth_shifted = teeth.shift(5)
    lips_shifted = lips.shift(3)
    
    # Alligator alignment signals
    bullish_alignment = (lips_shifted > teeth_shifted) & (teeth_shifted > jaw_shifted)
    bearish_alignment = (lips_shifted < teeth_shifted) & (teeth_shifted < jaw_shifted)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(bullish_alignment.iloc[i]) or np.isnan(bearish_alignment.iloc[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Bullish Alligator alignment, RSI oversold (<40), volume spike
        if (bullish_alignment.iloc[i] and 
            rsi_1d_aligned[i] < 40 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Bearish Alligator alignment, RSI overbought (>60), volume spike
        elif (bearish_alignment.iloc[i] and 
              rsi_1d_aligned[i] > 60 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Alligator alignment breaks or RSI returns to neutral (40-60)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (not bullish_alignment.iloc[i] or rsi_1d_aligned[i] >= 40)) or
               (signals[i-1] == -0.25 and (not bearish_alignment.iloc[i] or rsi_1d_aligned[i] <= 60)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_WilliamsAlligator_RSI1d_Volume"
timeframe = "4h"
leverage = 1.0