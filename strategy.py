#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# Long when: price breaks above Donchian(20) high AND 12h EMA(34) trending up AND volume > 1.5x 20-period average
# Short when: price breaks below Donchian(20) low AND 12h EMA(34) trending down AND volume > 1.5x 20-period average
# Exit when price crosses back through Donchian middle or volume drops below average
# Designed for ~25-35 trades/year per symbol with strong trend capture
name = "4h_Donchian_EMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_34_up = ema_12h_34 > np.roll(ema_12h_34, 1)
    ema_12h_34_up[0] = False
    ema_12h_34_up_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_34_up)
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    vol_confirm = vol_ratio > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_12h_34_up_aligned[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: break above Donchian high + uptrend + volume confirmation
            if price > donch_high[i] and ema_12h_34_up_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + downtrend + volume confirmation
            elif price < donch_low[i] and not ema_12h_34_up_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian mid OR volume confirmation lost
            if price < donch_mid[i] or not vol_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian mid OR volume confirmation lost
            if price > donch_mid[i] or not vol_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals