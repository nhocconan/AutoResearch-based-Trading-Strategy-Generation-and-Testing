#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Williams Alligator + 1d Trend + Volume Spike
# Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
# Combined with 1d trend filter and volume spikes, captures strong moves in both bull and bear markets.
# Works in bull markets via bullish alignment + uptrend, in bear via bearish alignment + downtrend.
# Volume spikes confirm institutional participation.
# Target: 19-50 trades/year (75-200 total over 4 years) for 4h timeframe.

name = "4h_williams_alligator_1d_trend_volume_v1"
timeframe = "4h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(20) for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Williams Alligator (13, 8, 5 SMAs with offsets)
    # Jaw: 13-period SMA, shifted 8 bars
    sma13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(sma13, 8)  # shift 8 bars forward
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMA, shifted 5 bars
    sma8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(sma8, 5)  # shift 5 bars forward
    teeth[:5] = np.nan
    
    # Lips: 5-period SMA, shifted 3 bars
    sma5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(sma5, 3)  # shift 3 bars forward
    lips[:3] = np.nan
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: alligator lines tangled or trend turns bearish
            # Tangled: jaws < teeth < lips (not bullish alignment)
            if not (jaw[i] > teeth[i] > lips[i]) or close[i] < ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: alligator lines tangled or trend turns bullish
            # Tangled: jaws > teeth > lips (not bearish alignment)
            if not (jaw[i] < teeth[i] < lips[i]) or close[i] > ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Bullish alignment + uptrend
                if jaw[i] > teeth[i] > lips[i] and close[i] > ema_20_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Bearish alignment + downtrend
                elif jaw[i] < teeth[i] < lips[i] and close[i] < ema_20_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals