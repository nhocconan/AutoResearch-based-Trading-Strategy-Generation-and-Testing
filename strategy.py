#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 Breakout with 1d EMA34 Trend Filter and Volume Spike Confirmation
# Camarilla pivot levels (R3/S3) act as strong intraday support/resistance levels
# Breakouts above R3 or below S3 with volume confirmation indicate institutional participation
# 1d EMA34 provides higher timeframe trend alignment to avoid counter-trend trades
# Volume spike (>2.0x 20-period average) confirms breakout validity
# Designed for low-frequency, high-conviction trades on 12h timeframe to minimize fee drag
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # 1d EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Need at least 1 day of data to calculate Camarilla levels
        if i < 24:  # Need 24*12h bars = 12 days? Actually need prior day's OHLC
            # For 12h timeframe, we need prior day's high/low/close
            # Assuming we have enough prior bars in the 12h data
            if i < 2:  # Need at least 2 bars (1 day) of prior data
                signals[i] = 0.0
                continue
        
        # Calculate Camarilla levels from prior day's OHLC
        # For 12h timeframe, prior day = 2 bars back
        prior_high = high[i-2]
        prior_low = low[i-2]
        prior_close = close[i-2]
        
        # Avoid division by zero
        if prior_high == prior_low:
            signals[i] = 0.0
            continue
            
        # Camarilla levels
        range_val = prior_high - prior_low
        camarilla_r3 = prior_close + range_val * 1.1 / 4
        camarilla_s3 = prior_close - range_val * 1.1 / 4
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        
        # Calculate 20-period average volume for confirmation
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            signals[i] = 0.0
            continue
            
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * vol_ma_20
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below camarilla S3 OR below 1d EMA34
            if curr_close < camarilla_s3 or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above camarilla R3 OR above 1d EMA34
            if curr_close > camarilla_r3 or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above camarilla R3 + above 1d EMA34 + volume confirmation
            if (curr_close > camarilla_r3 and 
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below camarilla S3 + below 1d EMA34 + volume confirmation
            elif (curr_close < camarilla_s3 and 
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals