#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1-day volume confirmation and 1-week trend filter.
# Uses 4-hour Donchian channel breakouts (20 periods) for entry, confirmed by 1-day volume spikes
# and filtered by 1-week EMA trend direction. Designed to capture strong momentum moves while
# avoiding counter-trend trades. Targets 20-50 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1-week data for EMA trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_length = 34
    ema_1w = pd.Series(close_1w).ewm(span=ema_length, adjust=False, min_periods=ema_length).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Load 1-day data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_length = 20
    vol_ma_1d = pd.Series(volume_1d).rolling(window=vol_ma_length, min_periods=vol_ma_length).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 4-hour Donchian channel
    high = prices['high'].values
    low = prices['low'].values
    donchian_length = 20
    upper_channel = pd.Series(high).rolling(window=donchian_length, min_periods=donchian_length).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_length, min_periods=donchian_length).min().values
    
    # Calculate 4-hour average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or 
            np.isnan(vol_ma_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol_4h = volume[i]
        vol_4h_ma = vol_ma_4h[i]
        vol_1d = vol_ma_1d_aligned[i]
        ema_trend = ema_1w_aligned[i]
        
        upper = upper_channel[i]
        lower = lower_channel[i]
        
        # Volume filter: current 4h volume > 1.5 * 4h average AND 1d volume > 1.5 * 1d average
        vol_4h_spike = vol_4h > 1.5 * vol_4h_ma
        vol_1d_spike = vol_1d > 1.5 * vol_ma_1d_aligned[i]
        volume_filter = vol_4h_spike and vol_1d_spike
        
        if position == 0:
            # Long entry: price breaks above Donchian upper channel with volume and uptrend
            if price > upper and volume_filter and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower channel with volume and downtrend
            elif price < lower and volume_filter and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on retracement to Donchian lower channel
                if price < lower:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on retracement to Donchian upper channel
                if price > upper:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DonchianBreakout_1dVolumeSpike_1wTrendFilter"
timeframe = "4h"
leverage = 1.0