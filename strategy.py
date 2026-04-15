#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Volume Spike + 1w Trend Filter
# Alligator: Jaw (EMA13, 8-shift), Teeth (EMA8, 5-shift), Lips (EMA5, 3-shift)
# Long when Lips > Teeth > Jaw (bullish alignment) + volume spike + 1w EMA50 uptrend
# Short when Lips < Teeth < Jaw (bearish alignment) + volume spike + 1w EMA50 downtrend
# Alligator signals trend presence and direction; volume confirms strength; 1w filter avoids counter-trend trades
# Uses discrete sizing (0.25) to limit overtrading and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Williams Alligator components
    # Jaw: EMA13 of median price, shifted 8 bars
    median_price = (high + low) / 2
    jaw_raw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan  # First 8 values invalid due to shift
    
    # Teeth: EMA8 of median price, shifted 5 bars
    teeth_raw = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan  # First 5 values invalid due to shift
    
    # Lips: EMA5 of median price, shifted 3 bars
    lips_raw = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan  # First 3 values invalid due to shift
    
    # Alligator alignment: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    bearish_alignment = (lips < teeth) & (teeth < jaw)
    
    # Volume confirmation: current > 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(13, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Bullish alignment + volume + 1w uptrend
        if (bullish_alignment[i] and volume[i] > vol_threshold[i] and 
            close[i] > ema_1w_aligned[i]):
            signals[i] = 0.25
        
        # Short: Bearish alignment + volume + 1w downtrend
        elif (bearish_alignment[i] and volume[i] > vol_threshold[i] and 
              close[i] < ema_1w_aligned[i]):
            signals[i] = -0.25
        
        # Exit: alignment breaks or trend fails
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (not bullish_alignment[i] or close[i] <= ema_1w_aligned[i])) or
               (signals[i-1] == -0.25 and (not bearish_alignment[i] or close[i] >= ema_1w_aligned[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WilliamsAlligator_Volume_Trend"
timeframe = "1d"
leverage = 1.0