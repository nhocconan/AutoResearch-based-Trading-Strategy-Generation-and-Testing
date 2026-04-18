#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h trend filter and volume confirmation.
# Long when: Close breaks above R1 and 12h EMA34 is rising, volume > 1.5x average.
# Short when: Close breaks below S1 and 12h EMA34 is falling, volume > 1.5x average.
# Exit when price returns to pivot (PP) or reverses at S2/R2.
# Uses Camarilla levels for mean reversion breakouts, higher timeframe for trend alignment,
# and volume to confirm institutional participation. Designed for ~25-40 trades/year per symbol.
name = "4h_Camarilla_R1_S1_Breakout_Volume_Trend"
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
    ema_12h_34_slope = ema_12h_34 - np.roll(ema_12h_34, 1)
    ema_12h_34_slope[0] = 0
    
    # Calculate 4-session average volume for confirmation
    vol_avg_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_12h_34_slope[i]) or np.isnan(vol_avg_4[i]) or 
            np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for current day using previous day's OHLC
        # Need previous day's data - we'll use daily resampling from 4h data
        # Group by date to get daily OHLC
        if i < 24:  # Need at least 24 hours (6 bars of 4h) for previous day
            signals[i] = 0.0
            continue
            
        # Find start of current trading day (assuming 00:00 UTC)
        # We'll use the last 6 bars (24h) to calculate daily OHLC
        lookback = min(i, 24)
        day_high = np.max(high[i-lookback+1:i+1])
        day_low = np.min(low[i-lookback+1:i+1])
        day_close = close[i-lookback+1]  # First bar of the period
        
        # Calculate Camarilla levels
        range_val = day_high - day_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla formulas
        pp = (day_high + day_low + day_close) / 3
        r1 = pp + (range_val * 1.1 / 12)
        s1 = pp - (range_val * 1.1 / 12)
        r2 = pp + (range_val * 1.1 / 6)
        s2 = pp - (range_val * 1.1 / 6)
        
        price = close[i]
        vol_ratio = volume[i] / (vol_avg_4[i] + 1e-10)
        ema_slope = ema_12h_34_slope[i]
        
        if position == 0:
            # Long: Price breaks above R1, 12h EMA trending up, volume confirmation
            if price > r1 and ema_slope > 0 and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1, 12h EMA trending down, volume confirmation
            elif price < s1 and ema_slope < 0 and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to pivot or reaches R2
            if price <= pp or price >= r2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to pivot or reaches S2
            if price >= pp or price <= s2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals