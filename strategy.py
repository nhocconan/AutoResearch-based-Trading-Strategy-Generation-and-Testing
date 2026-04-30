#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA50 trend filter and volume confirmation
# Williams %R measures overbought/oversold: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when %R crosses above -80 from below (oversold bounce) + price above 1d EMA50 + volume > 1.5x average
# Short when %R crosses below -20 from above (overbought rejection) + price below 1d EMA50 + volume > 1.5x average
# Uses 12h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Williams %R is effective in ranging markets and captures reversals in bear rallies
# Discrete position sizing: 0.25 for entries to limit fee drag
# Works in bull/bear regimes: trend filter prevents counter-trend trades

name = "12h_WilliamsR_1dEMA50_Volume_v1"
timeframe = "12h"
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
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    williams_r = np.where(rr != 0, ((highest_high - close) / rr) * -100, -50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 14, 20, 50)  # warmup for Williams %R (14), volume MA (20), EMA50 (50)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_prev_williams_r = williams_r[i-1]
        curr_volume_spike = volume_spike[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with trend filter and Williams %R reversal
            if curr_volume_spike:
                # Bullish: Williams %R crosses above -80 from below + price above 1d EMA50
                if curr_williams_r > -80 and curr_prev_williams_r <= -80 and curr_close > curr_ema_50_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Williams %R crosses below -20 from above + price below 1d EMA50
                elif curr_williams_r < -20 and curr_prev_williams_r >= -20 and curr_close < curr_ema_50_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum loss) or price below 1d EMA50 (trend break)
            if curr_williams_r < -50 or curr_close < curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum loss) or price above 1d EMA50 (trend break)
            if curr_williams_r > -50 or curr_close > curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals