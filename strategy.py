#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily range breakout with weekly trend filter and volume confirmation.
# Long when price breaks above weekly Donchian high (20-period) AND daily volume > 1.5x daily average volume AND price > weekly EMA34 (uptrend filter).
# Short when price breaks below weekly Donchian low (20-period) AND daily volume > 1.5x daily average volume AND price < weekly EMA34 (downtrend filter).
# Exit when price crosses back through the weekly Donchian midpoint.
# Uses weekly structure for trend, daily breakout for entry, volume for confirmation.
# Target: 10-25 trades/year per symbol (~40-100 total over 4 years).
name = "1d_WeeklyDonchian_Volume_EMA34"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and EMA34
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Donchian channels (20-period)
    high_roll_1w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_roll_1w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_1w = high_roll_1w
    donchian_low_1w = low_roll_1w
    donchian_mid_1w = (donchian_high_1w + donchian_low_1w) / 2
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_1w)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily average volume for confirmation (20-period)
    vol_ma_daily = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure EMA34 and Donchian are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma_daily[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_daily[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        mid = donchian_mid_aligned[i]
        ema34_val = ema34_aligned[i]
        
        # Volume confirmation: volume > 1.5x daily average
        vol_confirm = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: break above weekly Donchian high + volume + above weekly EMA34
            if price > upper and vol_confirm and price > ema34_val:
                signals[i] = 0.25
                position = 1
            # Short entry: break below weekly Donchian low + volume + below weekly EMA34
            elif price < lower and vol_confirm and price < ema34_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly Donchian midpoint
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly Donchian midpoint
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals