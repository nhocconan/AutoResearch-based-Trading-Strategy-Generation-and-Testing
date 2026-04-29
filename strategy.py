#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above R3 AND close > 1d EMA34 AND volume > 2.0x average
# Short when price breaks below S3 AND close < 1d EMA34 AND volume > 2.0x average
# Uses discrete sizing (0.25) and tight entry conditions to target 20-50 trades/year.
# Camarilla levels provide institutional support/resistance; 1d EMA34 filters trend; volume confirms conviction.
# Timeframe: 4h (primary), HTF: 1d for EMA34 trend and Camarilla calculation.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (using previous day's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), 
    #            S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    # But standard intraday Camarilla uses: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    # We'll use the more common 1.1 multiplier for R3/S3
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla R3 and S3 for 1d
    camarilla_r3_1d = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3_1d = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34 and Camarilla (need previous day data)
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_r3 = camarilla_r3_1d_aligned[i]
        curr_s3 = camarilla_s3_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price falls below R3 (breakout failed)
            # 2. Price crosses below 1d EMA34 (trend change)
            if (curr_close < curr_r3 or
                curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above S3 (breakdown failed)
            # 2. Price crosses above 1d EMA34 (trend change)
            if (curr_close > curr_s3 or
                curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND close > 1d EMA34 AND volume confirm
            if (curr_close > curr_r3 and
                curr_close > curr_ema_34_1d and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 AND close < 1d EMA34 AND volume confirm
            elif (curr_close < curr_s3 and
                  curr_close < curr_ema_34_1d and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals