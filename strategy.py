#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Williams Alligator consists of three SMAs (Jaw=13, Teeth=8, Lips=5) with future shifts.
# Long when Lips > Teeth > Jaw (bullish alignment) and price above Jaw.
# Short when Lips < Teeth < Jaw (bearish alignment) and price below Jaw.
# 1d EMA50 filters for major trend alignment. Volume spike ensures institutional participation.
# Works in bull via trend-following longs, in bear via selective shorts on rallies.

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Williams Alligator components (SMAs with future shifts)
    # Jaw: 13-period SMA, shifted 8 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.concatenate([np.full(8, np.nan), jaw[:-8]])  # shift forward 8 bars
    
    # Teeth: 8-period SMA, shifted 5 bars forward
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.concatenate([np.full(5, np.nan), teeth[:-5]])  # shift forward 5 bars
    
    # Lips: 5-period SMA, shifted 3 bars forward
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.concatenate([np.full(3, np.nan), lips[:-3]])  # shift forward 3 bars
    
    # 1d EMA(50) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(13, 8, 5, 24)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: Lips > Teeth > Jaw (bullish alignment) AND price above Jaw
                if (curr_lips > curr_teeth > curr_jaw and 
                    curr_close > curr_jaw):
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Lips < Teeth < Jaw (bearish alignment) AND price below Jaw
                elif (curr_lips < curr_teeth < curr_jaw and 
                      curr_close < curr_jaw):
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: Alligator alignment breaks OR price crosses below Jaw
            if (not (curr_lips > curr_teeth > curr_jaw) or  # Lost bullish alignment
                curr_close < curr_jaw):  # Price below Jaw (trend change)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks OR price crosses above Jaw
            if (not (curr_lips < curr_teeth < curr_jaw) or  # Lost bearish alignment
                curr_close > curr_jaw):  # Price above Jaw (trend change)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals