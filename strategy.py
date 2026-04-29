#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide high-probability reversal/breakout points
# 1w EMA34 establishes primary trend direction (bullish if price > EMA34, bearish if price < EMA34)
# Volume spike confirms institutional participation
# Works in bull markets (trend continuation via breakouts) and bear markets (mean reversion at extremes)
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe

name = "1d_Camarilla_R3S3_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
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
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla pivot levels from previous day
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    # Using previous day's OHLC to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # First bar has no previous day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range * 1.1 / 4
    s3 = prev_close - 1.1 * camarilla_range * 1.1 / 4
    
    # Calculate 20-period average volume for spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # 1w EMA34 and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1w = ema_34_1w_aligned[i]
        curr_r3 = r3[i]
        curr_s3 = s3[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_spike = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price breaks below S3 OR closes below 1w EMA34
            if curr_close < curr_s3 or curr_close < curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 OR closes above 1w EMA34
            if curr_close > curr_r3 or curr_close > curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND above 1w EMA34 AND volume spike
            if curr_close > curr_r3 and curr_close > curr_ema_1w and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 AND below 1w EMA34 AND volume spike
            elif curr_close < curr_s3 and curr_close < curr_ema_1w and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals