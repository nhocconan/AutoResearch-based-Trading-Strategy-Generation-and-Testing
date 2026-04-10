#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 2.0x 20-bar avg
# - Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 2.0x 20-bar avg
# - Exit when price crosses Donchian(10) midpoint (mean reversion)
# - Uses 1w EMA50 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-50 trades/year on 1d timeframe (80-200 total over 4 years)
# - Donchian breakouts work in both bull (trend continuation) and bear (mean reversion in ranges) markets

name = "1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period)
    highest_high = pd.Series(prices['high']).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(prices['low']).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    donchian_mid_10 = (pd.Series(prices['high']).rolling(window=10, min_periods=10).max().values + 
                       pd.Series(prices['low']).rolling(window=10, min_periods=10).min().values) / 2.0
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1w data
    c_1w = df_1w['close'].values
    c_1w_aligned = align_htf_to_ltf(prices, df_1w, c_1w)
    
    # Pre-compute 1w EMA(50) for trend filter
    ema50_1w = pd.Series(c_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(c_1w_aligned[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(donchian_mid_10[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian(20) high AND in 1w uptrend with volume spike
            if (prices['close'].iloc[i] > highest_high[i] and 
                prices['close'].iloc[i] > ema50_1w_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian(20) low AND in 1w downtrend with volume spike
            elif (prices['close'].iloc[i] < lowest_low[i] and 
                  prices['close'].iloc[i] < ema50_1w_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price crosses Donchian(10) midpoint (mean reversion)
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] < donchian_mid_10[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] > donchian_mid_10[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals