#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume spike
# Long when price breaks above Donchian upper (20) AND 1d EMA34 trend up AND volume spike
# Short when price breaks below Donchian lower (20) AND 1d EMA34 trend down AND volume spike
# Exit when price returns to Donchian middle or trend weakens
# Designed for low trade frequency (~15-30/year) with edge in trending markets
# Works in both bull (breakouts up) and bear (breakouts down) markets

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA for 1d trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channels (20-period) on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper band: highest high of last 20 periods
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        donch_mid_val = donch_mid[i]
        ema_trend = ema_34_1d_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND 1d EMA trending up AND volume spike
            if price > donch_high_val and ema_trend > donch_mid_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND 1d EMA trending down AND volume spike
            elif price < donch_low_val and ema_trend < donch_mid_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to Donchian middle or trend weakens
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to middle or trend turns down
                if price < donch_mid_val or ema_trend < donch_mid_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to middle or trend turns up
                if price > donch_mid_val or ema_trend > donch_mid_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0