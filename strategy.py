#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA200 trend filter and volume spike confirmation.
# Donchian channel breakouts capture strong momentum moves. Combined with daily EMA200 trend filter
# to only trade in the direction of the long-term trend and volume spikes (>2x 20-period average)
# to confirm institutional participation. This reduces false signals and focuses on high-probability
# breakouts. Designed for low trade frequency (~15-35/year) to minimize fee decay. Works in both
# bull and bear markets by following the daily trend direction.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for EMA200 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 200-period EMA on daily close for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily EMA200 to 1h timeframe (waits for daily bar to close)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 20-period Donchian channel on 1h high/low
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_200_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema_trend = ema_200_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + above daily EMA200 + volume spike
            if price > upper and price > ema_trend and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Donchian low + below daily EMA200 + volume spike
            elif price < lower and price < ema_trend and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions: price breaks back through the opposite Donchian band
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below Donchian low
                if price < lower:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above Donchian high
                if price > upper:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_DonchianBreakout_1dEMA200_Volume"
timeframe = "1h"
leverage = 1.0