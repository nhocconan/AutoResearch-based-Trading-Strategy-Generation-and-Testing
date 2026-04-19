#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ehlers Fisher Transform with daily trend filter and volume confirmation.
# Fisher Transform (period=9) catches reversals in overbought/oversold conditions.
# Long when Fisher crosses above -1.5 AND price above daily EMA34 AND volume spike (>1.8x average).
# Short when Fisher crosses below +1.5 AND price below daily EMA34 AND volume spike.
# Uses daily EMA34 as trend filter to avoid counter-trend trades.
# Volume confirmation ensures reversals have institutional participation.
# Target: 12-37 trades/year per symbol (~50-150 total over 4 years).
name = "6h_FisherTransform_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA34 calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on daily close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe (wait for daily close)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Fisher Transform on 6h price (period=9)
    # Fisher Transform formula: 0.5 * ln((1+X)/(1-X)) where X is normalized price
    high_low = high - low
    highest_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    lowest_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1, range_hl)
    
    # Normalize price to [-1, 1] range
    value = 2 * ((close - lowest_low) / range_hl) - 1
    value = np.clip(value, -0.999, 0.999)  # Prevent log(0)
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + value) / (1 - value))
    # Smooth with 2-period EMA as per Ehlers
    fisher_smooth = pd.Series(fisher).ewm(span=2, adjust=False).values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 9)  # Need volume MA, EMA, and Fisher data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(fisher_smooth[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_34_aligned[i]
        fisher_val = fisher_smooth[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.8 * vol_ma
        
        # Fisher thresholds for reversal signals
        fisher_long_signal = fisher_val > -1.5
        fisher_short_signal = fisher_val < 1.5
        
        if position == 0:
            # Enter long: Fisher crosses above -1.5 AND price above daily EMA34
            if (i > start_idx and 
                fisher_smooth[i-1] <= -1.5 and fisher_val > -1.5 and  # Cross above -1.5
                price > ema_trend and volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Enter short: Fisher crosses below +1.5 AND price below daily EMA34
            elif (i > start_idx and 
                  fisher_smooth[i-1] >= 1.5 and fisher_val < 1.5 and  # Cross below 1.5
                  price < ema_trend and volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when Fisher crosses below +1.5 or price below daily EMA34
            if (fisher_val < 1.5 or price < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Fisher crosses above -1.5 or price above daily EMA34
            if (fisher_val > -1.5 or price > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals