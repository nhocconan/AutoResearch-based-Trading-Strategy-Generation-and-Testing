#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w trend filter and volume confirmation
# - Long when price breaks above 20-period Donchian high with volume > 1.5x average AND 1w close > 1w EMA50
# - Short when price breaks below 20-period Donchian low with volume > 1.5x average AND 1w close < 1w EMA50
# - Exit when price retreats to midpoint of Donchian channel OR volume drops below 0.7x average
# - Uses 1w trend filter to avoid counter-trend trades in bear markets (2025+)
# - Moderate volume threshold (1.5x) reduces false breakouts while maintaining sufficient trade frequency
# - Target: 30-80 trades over 4 years (7-20/year) to minimize fee drag while capturing major trends
# - Discrete position sizing (0.25) to reduce churn from minor signal fluctuations

name = "1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute Donchian channel (20-period) on 1d data
    highest_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    lowest_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > Donchian high with volume spike AND 1w uptrend
            if (prices['close'].iloc[i] > highest_20[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < Donchian low with volume spike AND 1w downtrend
            elif (prices['close'].iloc[i] < lowest_20[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0  # Stay flat
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retreats to Donchian midpoint
            # 2. Volume drops below 0.7x average (loss of momentum)
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < donchian_mid[i] or 
                    vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > donchian_mid[i] or 
                    vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals