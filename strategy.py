#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action combined with 12h momentum and volume confirmation
# Uses price crossing above/below 12h EMA34 with volume spike (>2x average) for entry
# Exit when price crosses back or volume drops
# Designed to capture strong momentum moves in both bull and bear markets
# Target: 20-40 trades/year per symbol (~80-160 total over 4 years)

name = "4h_12hEMA34_VolumeBreakout"
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
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: current volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need volume MA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_30[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_34_12h_aligned[i]
        vol_ma = vol_ma_30[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Enter long: price crosses above 12h EMA34 with volume
            if price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below 12h EMA34 with volume
            elif price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below 12h EMA34
            if price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above 12h EMA34
            if price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals