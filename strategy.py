#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1-week Donchian breakout with 1-day volume and trend filter
# Long when price breaks above weekly Donchian upper channel in uptrend (close > 1d EMA34) with volume spike (>2x 20-period avg)
# Short when price breaks below weekly Donchian lower channel in downtrend (close < 1d EMA34) with volume spike
# Exit when price retouches the opposite channel or trend reverses
# Designed for low trade frequency (~15-30/year) to minimize fee drain. Weekly Donchian provides structure,
# 1d EMA34 filters trend direction, volume confirms breakout strength. Works in bull/bear by combining
# long-term structure with intermediate trend and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data for Donchian channel (5 trading days per week = 5 periods)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 5-period Donchian channel on weekly data
    # Upper = max(high over last 5 weeks), Lower = min(low over last 5 weeks)
    high_max_5w = pd.Series(high_1w).rolling(window=5, min_periods=5).max().values
    low_min_5w = pd.Series(low_1w).rolling(window=5, min_periods=5).min().values
    
    # Load 1d data for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period average volume for volume spike detection
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly Donchian levels and 1d indicators to 12h timeframe
    upper_chan_aligned = align_htf_to_ltf(prices, df_1w, high_max_5w)
    lower_chan_aligned = align_htf_to_ltf(prices, df_1w, low_min_5w)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_chan_aligned[i]) or 
            np.isnan(lower_chan_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = prices['volume'].iloc[i]
        upper_val = upper_chan_aligned[i]
        lower_val = lower_chan_aligned[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma_val
        
        if position == 0:
            # Long conditions: price breaks above weekly upper channel + uptrend + volume spike
            if price > upper_val and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below weekly lower channel + downtrend + volume spike
            elif price < lower_val and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price retouches opposite channel or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price touches or crosses lower channel or trend turns down
                if price < lower_val or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price touches or crosses upper channel or trend turns up
                if price > upper_val or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_1wDonchian5_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0