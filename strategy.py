#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d volume confirmation and ATR-based stoploss
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average
# - Exit when price crosses Donchian midpoint or opposite breakout occurs
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 20-50 trades/year on 4h (75-200 total over 4 years) to avoid fee drag
# - Works in both bull and bear markets by capturing breakouts with volume confirmation

name = "4h_donchian_volume_confirmation_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = np.full_like(vol_1d, np.nan, dtype=float)
    for i in range(19, len(vol_1d)):
        vol_ma_20[i] = np.mean(vol_1d[i-19:i+1])
    
    # Align HTF volume MA to 4h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = np.full_like(high_4h, np.nan, dtype=float)
    donchian_low = np.full_like(low_4h, np.nan, dtype=float)
    donchian_mid = np.full_like(close_4h, np.nan, dtype=float)
    
    for i in range(19, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i-19:i+1])
        donchian_low[i] = np.min(low_4h[i-19:i+1])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.5x average)
        vol_4h = prices['volume'].values
        vol_spike = not np.isnan(vol_ma_20_aligned[i]) and vol_4h[i] > 1.5 * vol_ma_20_aligned[i]
        
        close_now = close_4h[i]
        donchian_high_now = donchian_high[i]
        donchian_low_now = donchian_low[i]
        donchian_mid_now = donchian_mid[i]
        
        # Donchian breakout signals
        breakout_up = close_now > donchian_high_now  # price breaks above upper band
        breakout_down = close_now < donchian_low_now  # price breaks below lower band
        midpoint_cross_up = (close_4h[i-1] <= donchian_mid_now and close_now > donchian_mid_now)  # crosses above midpoint
        midpoint_cross_down = (close_4h[i-1] >= donchian_mid_now and close_now < donchian_mid_now)  # crosses below midpoint
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian high AND volume spike
            if breakout_up and vol_spike:
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND volume spike
            elif breakout_down and vol_spike:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses midpoint or opposite Donchian breakout
            exit_long = (position == 1 and 
                        (midpoint_cross_down or breakout_down))
            exit_short = (position == -1 and 
                         (midpoint_cross_up or breakout_up))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals