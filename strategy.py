#!/usr/bin/env python3
# 12H_DONCHIAN20_VOLUME_CONFIRMATION_1D_TREND_FILTER
# Hypothesis: Donchian breakouts capture strong momentum moves; volume confirmation filters false breakouts;
# 1D trend filter avoids counter-trend trades. Works in bull markets (breakout continuations) and bear markets
# (sharp reversals after volatility spikes). Target: 15-30 trades/year on 12h timeframe.

name = "12H_DONCHIAN20_VOLUME_CONFIRMATION_1D_TREND_FILTER"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Average volume for confirmation (20-period)
    vol_avg = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align to 12h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(ema34_aligned[i]) or np.isnan(vol_avg_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channels (20-period)
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
            
            # Volume confirmation: current volume > 1.5x average volume
            vol_confirmed = volume[i] > 1.5 * vol_avg_aligned[i]
            
            if position == 0:
                # LONG: Price breaks above Donchian high in uptrend with volume
                if (close[i] > highest_high and 
                    close[i] > ema34_aligned[i] and 
                    vol_confirmed):
                    signals[i] = 0.25
                    position = 1
                # SHORT: Price breaks below Donchian low in downtrend with volume
                elif (close[i] < lowest_low and 
                      close[i] < ema34_aligned[i] and 
                      vol_confirmed):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: Price breaks below Donchian low or trend reversal
                if (close[i] < lowest_low or 
                    close[i] <= ema34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # EXIT SHORT: Price breaks above Donchian high or trend reversal
                if (close[i] > highest_high or 
                    close[i] >= ema34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals