#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Uses tight volume threshold (2.5x average) to limit trades to ~100 total over 4 years.
# Only enters when price breaks 12h Camarilla R3 (for longs) or S3 (for shorts) level with volume confirmation and 1d EMA34 trend alignment.
# Designed for low trade frequency (<150 total 12h trades) to avoid fee drag. Works in bull/bear via 1d EMA34 trend filter.

name = "12h_Camarilla_R3S3_1dEMA34_VolumeConfirm_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        # Calculate 12h Camarilla levels using only completed 12h bars
        # We need the previous completed 12h bar's OHLC
        if i >= 1:
            # Get the previous completed 12h bar (index i-1)
            phigh = high[i-1]
            plow = low[i-1]
            pclose = close[i-1]
            
            # Calculate Camarilla levels
            range_val = phigh - plow
            if range_val <= 0:
                r3 = s3 = np.nan
            else:
                r3 = pclose + (range_val * 1.1 / 4)  # R3 level
                s3 = pclose - (range_val * 1.1 / 4)  # S3 level
        else:
            r3 = s3 = np.nan
        
        # Volume confirmation: volume > 2.5x 20-period average (tight threshold to reduce trades)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.5 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, 1d EMA34 uptrend, volume spike confirmation
            if (not np.isnan(r3) and not np.isnan(s3) and
                curr_close > r3 and 
                curr_close > curr_ema_34_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3, 1d EMA34 downtrend, volume spike confirmation
            elif (not np.isnan(r3) and not np.isnan(s3) and
                  curr_close < s3 and 
                  curr_close < curr_ema_34_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when price closes below 1d EMA34 (trend reversal)
            if curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price closes above 1d EMA34 (trend reversal)
            if curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals