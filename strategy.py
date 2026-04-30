#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (>1.8x average)
# Camarilla pivot levels provide institutional support/resistance. Breakout at R3/S3 with volume
# and 1d trend filter avoids false breakouts. Works in bull/bear via trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.30

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Volume_v1"
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
    
    # Calculate daily pivot points for Camarilla levels
    # Need previous day's OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_day_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_day_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_day_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_r3 = prev_day_close + (prev_day_high - prev_day_low) * 1.1 / 4
    camarilla_s3 = prev_day_close - (prev_day_high - prev_day_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 1.8x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 34)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_spike = volume_spike[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with trend filter and Camarilla breakout
            if curr_volume_spike:
                # Bullish: price breaks above Camarilla R3 + price above 1d EMA34
                if curr_high > curr_r3 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.30
                    position = 1
                # Bearish: price breaks below Camarilla S3 + price below 1d EMA34
                elif curr_low < curr_s3 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 (trend reversal)
            if curr_low < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 (trend reversal)
            if curr_high > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals