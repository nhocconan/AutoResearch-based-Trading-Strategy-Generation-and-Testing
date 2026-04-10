#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR filter and volume confirmation
# - Long when price breaks above 20-bar Donchian high AND ATR(14) > 20-bar ATR mean AND volume > 1.5x 20-bar avg
# - Short when price breaks below 20-bar Donchian low AND ATR(14) > 20-bar ATR mean AND volume > 1.5x 20-bar avg
# - Exit when price crosses 10-bar EMA in opposite direction (trend-following exit)
# - Uses 1d ATR filter to avoid low-volatility breakouts that fail
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-35 trades/year on 4h timeframe (80-140 total over 4 years)
# - Donchian breakouts work in both bull (trend continuation) and bear (trend acceleration) markets
# - ATR filter ensures breakouts occur during sufficient volatility, reducing false signals

name = "4h_1d_donchian_breakout_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period)
    donchian_high = prices['high'].rolling(window=20, min_periods=20).max().values
    donchian_low = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR(14) for volatility filter
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute 20-period ATR mean for volatility regime filter
    atr_20_mean = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute 10-period EMA for exit signal
    ema10 = pd.Series(prices['close'].values).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_20_mean[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(ema10[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND high volatility AND volume spike
            if (prices['close'].iloc[i] > donchian_high[i] and 
                atr[i] > atr_20_mean[i] and  # high volatility regime
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND high volatility AND volume spike
            elif (prices['close'].iloc[i] < donchian_low[i] and 
                  atr[i] > atr_20_mean[i] and  # high volatility regime
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit based on EMA crossover
            # Exit when price crosses 10-period EMA in opposite direction
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] < ema10[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] > ema10[i]:
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