#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d trend filter
# - Long when price breaks above 20-period high AND volume > 2.0x 20-bar average AND 1d close > 1d EMA50
# - Short when price breaks below 20-period low AND volume > 2.0x 20-bar average AND 1d close < 1d EMA50
# - Exit when price returns to 20-period midpoint or opposite Donchian level is touched
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Targets ~20-30 trades/year (80-120 total over 4 years) to avoid fee drag
# - Donchian channels provide clear structure in both trending and ranging markets
# - Volume confirmation ensures breakout validity and reduces false signals
# - 1d EMA50 filter ensures alignment with higher timeframe trend for better generalization

name = "4h_1d_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 20-period Donchian channels for 4h data
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        current_price = prices['close'].iloc[i]
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above 20-period high with volume spike and 1d uptrend
            if (current_price > high_20[i] and 
                vol_spike.iloc[i] and 
                current_price > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below 20-period low with volume spike and 1d downtrend
            elif (current_price < low_20[i] and 
                  vol_spike.iloc[i] and 
                  current_price < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price returns to 20-period midpoint
            # 2. Opposite Donchian level is touched (long exits at low_20, short exits at high_20)
            if position == 1:
                if current_price <= mid_20[i] or current_price < low_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if current_price >= mid_20[i] or current_price > high_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals