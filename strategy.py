#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R3 with 1d EMA34 uptrend and volume > 2.0x 20-bar average.
# Short when price breaks below S3 with 1d EMA34 downtrend and volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 12h timeframe to avoid overtrading.
# Works in bull (buy breakouts) and bear (sell breakdowns) via trend filter.

name = "12h_Camarilla_R3_S3_1dEMA34_VolumeSpike_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for EMA34 and Camarilla calculation
    
    for i in range(start_idx, n):
        # Session filter: 00-23 UTC (trade all sessions for 12h timeframe)
        hour = hours[i]
        
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        # Calculate Camarilla levels from previous 20 12h bars (typical pivot lookback)
        # Use bars i-20 to i-1 for prior period calculation
        if i-20 < start_idx:
            signals[i] = 0.0
            continue
            
        lookback_high = np.max(high[i-20:i])  # highest high of last 20 bars
        lookback_low = np.min(low[i-20:i])    # lowest low of last 20 bars
        lookback_close = close[i-1]           # previous close for pivot calculation
        
        if lookback_high <= lookback_low:
            signals[i] = 0.0
            continue
            
        # Camarilla levels calculation
        range_val = lookback_high - lookback_low
        r3 = lookback_close + range_val * 1.1 / 4
        s3 = lookback_close - range_val * 1.1 / 4
        
        # Volume confirmation: current 12h volume > 2.0x 20-period average
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        vol_ma = np.mean(volume[i-20:i])  # 20-period simple moving average
        if vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (vol_ma * 2.0)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 AND price > 1d EMA34 AND volume confirmation
            if (curr_close > r3 and 
                curr_close > curr_ema_34_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND price < 1d EMA34 AND volume confirmation
            elif (curr_close < s3 and 
                  curr_close < curr_ema_34_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below S3 (reversal) OR price < 1d EMA34 (trend violation)
            if (curr_close < s3 or 
                curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above R3 (reversal) OR price > 1d EMA34 (trend violation)
            if (curr_close > r3 or 
                curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals