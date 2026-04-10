#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w trend filter and 1d volume spike
# - Primary: 6h timeframe for balance of signal frequency and fee drag
# - HTF: 1w for trend direction (price > EMA50), 1d for volume confirmation
# - Long: Price breaks above 6h Donchian H20 + 1w close > EMA50 + 1d volume > 1.5x 20-period MA
# - Short: Price breaks below 6h Donchian L20 + 1w close < EMA50 + 1d volume > 1.5x 20-period MA
# - Exit: Price reverts to 6h Donchian midpoint (mean reversion) or breaks opposite H20/L20
# - Position sizing: 0.25 (discrete level)
# - Target: 80-150 total trades over 4 years (20-37/year) - within 6h sweet spot
# - Works in bull/bear: Donchian breakouts capture trends, weekly EMA filter avoids counter-trend trades

name = "6h_1w_1d_donchian_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h Donchian Channel (20-period)
    high_roll = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_h20 = high_roll
    donchian_l20 = low_roll
    donchian_mid = (donchian_h20 + donchian_l20) / 2.0
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_h20[i]) or np.isnan(donchian_l20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w close > EMA50 for long, < EMA50 for short
        uptrend = close_1w[i] > ema_50_1w_aligned[i]
        downtrend = close_1w[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian H20 + uptrend + volume spike
            if (close_6h[i] > donchian_h20[i] and uptrend and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian L20 + downtrend + volume spike
            elif (close_6h[i] < donchian_l20[i] and downtrend and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Donchian midpoint (mean reversion)
            # 2. Price breaks opposite Donchian level (take profit/stop)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_6h[i] < donchian_mid[i] or  # Reverted to midpoint
                    close_6h[i] > donchian_h20[i]     # Break above H20 (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_6h[i] > donchian_mid[i] or  # Reverted to midpoint
                    close_6h[i] < donchian_l20[i]     # Break below L20 (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals