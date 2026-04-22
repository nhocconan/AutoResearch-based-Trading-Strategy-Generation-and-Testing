#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout (20-period) with 12h EMA trend filter and volume spike confirmation.
# Targets strong momentum moves in trending markets while avoiding false breakouts in choppy conditions.
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for 20-50 trades/year on 4h.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for EMA trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA on 12h
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h Donchian channels (20-period high/low)
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema = ema_12h_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above Donchian high with uptrend (price > 12h EMA) and volume spike
            if price > donchian_high[i] and price > ema and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with downtrend (price < 12h EMA) and volume spike
            elif price < donchian_low[i] and price < ema and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on retracement to Donchian low (trailing stop)
                if price < donchian_low[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on retracement to Donchian high (trailing stop)
                if price > donchian_high[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0