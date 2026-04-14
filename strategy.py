#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator (13,8,5 SMAs) with 1d trend filter (EMA50) and volume confirmation.
# Alligator jaws (13 SMA), teeth (8 SMA), lips (5 SMA) define trend: 
#   Bullish: lips > teeth > jaws, Bearish: jaws > teeth > lips.
# 1d EMA50 provides higher timeframe trend bias to avoid counter-trend trades.
# 4x average volume confirms institutional participation in breakouts.
# Works in bull/bear as 1d EMA adapts to trend and Alligator catches reversals.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    ema_len = 50
    if len(df_1d) < ema_len:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Williams Alligator on 4h: SMAs of median price
    # Jaw: 13-period SMA, Teeth: 8-period SMA, Lips: 5-period SMA
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Volume confirmation: 4x average volume (institutional participation)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(60, 13, 8, 5, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: bullish (lips > teeth > jaws) or bearish (jaws > teeth > lips)
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alignment = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Trend filter: price relative to 1d EMA50
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 4x average
        volume_confirmed = volume[i] > 4.0 * vol_ma[i]
        
        if position == 0:
            # Enter long: Alligator bullish + above 1d EMA + volume
            if bullish_alignment and above_ema and volume_confirmed:
                position = 1
                signals[i] = position_size
            # Enter short: Alligator bearish + below 1d EMA + volume
            elif bearish_alignment and below_ema and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator turns bearish OR price crosses below 1d EMA
            if not bullish_alignment or below_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator turns bullish OR price crosses above 1d EMA
            if not bearish_alignment or above_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Alligator_EMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0