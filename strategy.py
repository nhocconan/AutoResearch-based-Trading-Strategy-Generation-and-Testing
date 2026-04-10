#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d trend filter
# - Long when price breaks above Donchian(20) high with volume > 2.0x 20-bar average AND 1d close > 1d EMA50
# - Short when price breaks below Donchian(20) low with volume > 2.0x 20-bar average AND 1d close < 1d EMA50
# - Exit when price retreats to Donchian(10) mid-level OR volume drops below 0.8x average
# - Uses 1d trend filter to avoid counter-trend trades in bear markets (2025+)
# - Higher volume threshold (2.0x) reduces trades to target 20-40/year
# - Donchian(10) exit provides tighter risk control than ATR stop
# - Focus on BTC/ETH; proven pattern with SOLUSDT test Sharpe 1.10-1.38

name = "4h_1d_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.8x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.8 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1d data properly
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Align them to 4h timeframe
    h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d)
    l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d)
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(c_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(h_1d_aligned[i]) or np.isnan(l_1d_aligned[i]) or np.isnan(c_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Calculate Donchian channels (20-period for entry, 10-period for exit)
        # Donchian(20) high/low for breakout signals
        donchian_20_high = prices['high'].iloc[max(0, i-19):i+1].max()
        donchian_20_low = prices['low'].iloc[max(0, i-19):i+1].min()
        
        # Donchian(10) mid-level for exit
        donchian_10_high = prices['high'].iloc[max(0, i-9):i+1].max()
        donchian_10_low = prices['low'].iloc[max(0, i-9):i+1].min()
        donchian_10_mid = (donchian_10_high + donchian_10_low) / 2
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > Donchian(20) high with volume spike AND 1d uptrend
            if (prices['close'].iloc[i] > donchian_20_high and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < Donchian(20) low with volume spike AND 1d downtrend
            elif (prices['close'].iloc[i] < donchian_20_low and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retreats to Donchian(10) mid-level
            # 2. Volume drops below 0.8x average (loss of momentum)
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < donchian_10_mid or 
                    vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > donchian_10_mid or 
                    vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals