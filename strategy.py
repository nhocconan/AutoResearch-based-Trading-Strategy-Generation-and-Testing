#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation.
# Long when price breaks above R3, 1d EMA34 uptrend, and volume > 2x 20-bar avg.
# Short when price breaks below S3, 1d EMA34 downtrend, and volume > 2x 20-bar avg.
# Exit when price retests the pivot point (mean reversion).
# Camarilla levels provide precise intraday support/resistance; 1d EMA34 filters trend direction.
# Volume confirmation reduces false breakouts. Timeframe: 12h as per experiment guidelines.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Need at least 2 bars to calculate pivot points (previous day's H/L/C)
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Use previous bar's high/low/close for today's Camarilla calculation
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate Camarilla pivot levels for today
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_hl = prev_high - prev_low
        
        # Resistance levels
        r3 = pivot + (range_hl * 1.1 / 2.0)  # R3 = pivot + 1.1*(H-L)/2
        # Support levels
        s3 = pivot - (range_hl * 1.1 / 2.0)  # S3 = pivot - 1.1*(H-L)/2
        
        # Volume confirmation: volume > 2x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_confirm = False
        
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, 1d EMA34 uptrend, volume spike
            if (curr_close > r3 and 
                curr_close > curr_ema_34_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, 1d EMA34 downtrend, volume spike
            elif (curr_close < s3 and 
                  curr_close < curr_ema_34_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price retests pivot point (mean reversion)
            if curr_close <= pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price retests pivot point (mean reversion)
            if curr_close >= pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals