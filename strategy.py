#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
# Donchian breakouts capture momentum bursts; weekly EMA50 filters for primary trend direction;
# volume spikes (>2x 20-period average) confirm institutional interest. Designed for low trade
# frequency (~15-25/year) to minimize fee decay. Works in bull markets via breakouts and in
# bear markets via breakdowns with trend filter preventing counter-trend trades.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data for Donchian calculation (once before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 20-period Donchian channels on 12h high/low
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50-period EMA on 12h close for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to primary timeframe (waits for 12h bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load 1w data for additional trend confirmation (optional but recommended)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ema_12h = ema_50_aligned[i]
        ema_1w = ema_50_1w_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        # Trend filter: both 12h and 1w EMA50 agree on direction
        bullish_trend = price > ema_12h and price > ema_1w
        bearish_trend = price < ema_12h and price < ema_1w
        
        if position == 0:
            # Long: price breaks above Donchian high + bullish trend + volume spike
            if price > upper and bullish_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + bearish trend + volume spike
            elif price < lower and bearish_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below Donchian low or trend turns bearish
                if price < lower or not bullish_trend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above Donchian high or trend turns bullish
                if price > upper or not bearish_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0