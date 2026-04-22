#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + Daily VWAP Trend with Volume Spike
# Long when Williams %R < -80 (oversold) + price > daily VWAP + volume spike
# Short when Williams %R > -20 (overbought) + price < daily VWAP + volume spike
# Exit when Williams %R returns to -50 level or trend reverses
# Williams %R identifies mean reversion extremes; VWAP provides institutional trend filter
# Volume spike confirms institutional participation. Designed for 15-25 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Williams %R and VWAP
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    volume_daily = df_daily['volume'].values
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_daily).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_daily).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_daily) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Calculate daily VWAP
    # VWAP = Cumulative (Price * Volume) / Cumulative Volume
    typical_price = (high_daily + low_daily + close_daily) / 3
    pv = typical_price * volume_daily
    cum_pv = np.nancumsum(pv)
    cum_volume = np.nancumsum(volume_daily)
    vwap = np.where(cum_volume != 0, cum_pv / cum_volume, typical_price)
    
    # Align Williams %R and VWAP to 4h timeframe (previous day's values)
    williams_r_aligned = align_htf_to_ltf(prices, df_daily, williams_r)
    vwap_aligned = align_htf_to_ltf(prices, df_daily, vwap)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(vwap_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        wr = williams_r_aligned[i]
        vwap_val = vwap_aligned[i]
        
        # Volume filter: current volume > 2.2 * 20-period average
        vol_spike = vol > 2.2 * vol_ma
        
        if position == 0:
            # Long conditions: Williams %R oversold (< -80) + price > VWAP + volume spike
            if wr < -80 and price > vwap_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought (> -20) + price < VWAP + volume spike
            elif wr > -20 and price < vwap_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R returns to -50 level or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R returns to -50 or price < VWAP
                if wr >= -50 or price < vwap_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R returns to -50 or price > VWAP
                if wr <= -50 or price > vwap_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_VWAP_Trend_Volume"
timeframe = "4h"
leverage = 1.0