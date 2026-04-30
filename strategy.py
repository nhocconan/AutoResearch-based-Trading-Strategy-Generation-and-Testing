#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 12h EMA50 trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions - extremes (>80 or <20) signal potential reversals
# 12h EMA50 provides intermediate trend filter to avoid counter-trend trades in ranging markets
# Volume confirmation (>1.4x average) ensures legitimacy of reversal signals
# Works in bull/bear: %R extremes occur in all markets, volume confirms legitimacy, trend filter reduces whipsaws
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity with fee control

name = "6h_WilliamsR_Extreme_12hEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.4x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.4 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on Williams %R extremes with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish reversal: %R below -80 (oversold) + price above 12h EMA50
                if curr_williams_r < -80 and curr_close > curr_ema_50_12h:
                    signals[i] = 0.25
                    position = 1
                # Bearish reversal: %R above -20 (overbought) + price below 12h EMA50
                elif curr_williams_r > -20 and curr_close < curr_ema_50_12h:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: %R crosses above -50 (momentum weakening) or opposite extreme
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: %R crosses below -50 (momentum weakening) or opposite extreme
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals