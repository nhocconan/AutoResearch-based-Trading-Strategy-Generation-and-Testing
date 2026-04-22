#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Long when price breaks above Donchian(20) high + weekly pivot direction = bullish + volume spike
# Short when price breaks below Donchian(20) low + weekly pivot direction = bearish + volume spike
# Weekly pivot direction based on price relative to weekly VWAP (bullish if price > weekly VWAP)
# Volume spike: current volume > 2.0 * 20-period average volume
# Designed for low trade frequency (~15-30/year) to minimize fee drain.
# Works in bull/bear by combining breakout momentum with weekly trend filter and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot direction filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly VWAP for trend direction
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    vwap_num_1w = typical_price_1w * volume_1w
    vwap_den_1w = volume_1w
    vwap_cum_num = np.nancumsum(vwap_num_1w)
    vwap_cum_den = np.nancumsum(vwap_den_1w)
    vwap_1w = np.where(vwap_cum_den != 0, vwap_cum_num / vwap_cum_den, close_1w)
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # Calculate Donchian(20) channels on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vwap_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        vwap_val = vwap_1w_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: breakout above Donchian high + bullish weekly trend + volume spike
            if price > donchian_high[i] and price > vwap_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: breakout below Donchian low + bearish weekly trend + volume spike
            elif price < donchian_low[i] and price < vwap_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price reverses back into Donchian channel or trend changes
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price falls back below Donchian low or weekly trend turns bearish
                if price < donchian_low[i] or price < vwap_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price rises back above Donchian high or weekly trend turns bullish
                if price > donchian_high[i] or price > vwap_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_WeeklyVWAP_Volume"
timeframe = "6h"
leverage = 1.0