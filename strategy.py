#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot breakouts with 1d EMA34 trend filter and volume spike confirmation.
# Weekly Camarilla levels provide strong institutional support/resistance; 1d EMA34 filters for intermediate-term trend alignment;
# Volume spikes confirm institutional participation in breakouts. Designed to work in both bull (breakout continuation) and bear (fade at extreme levels) markets.
# Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag while capturing significant moves.

name = "6h_WeeklyCamarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 10 or len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from previous week
    prev_weekly_close = np.concatenate([[np.nan], df_1w['close'].values[:-1]])
    prev_weekly_high = np.concatenate([[np.nan], df_1w['high'].values[:-1]])
    prev_weekly_low = np.concatenate([[np.nan], df_1w['low'].values[:-1]])
    
    camarilla_r3_weekly = prev_weekly_close + 1.0 * (prev_weekly_high - prev_weekly_low)
    camarilla_s3_weekly = prev_weekly_close - 1.0 * (prev_weekly_high - prev_weekly_low)
    
    camarilla_r3_weekly_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_weekly)
    camarilla_s3_weekly_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_weekly)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 20)  # warmup for EMA34 and volume
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(camarilla_r3_weekly_aligned[i]) or 
            np.isnan(camarilla_s3_weekly_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_r3 = camarilla_r3_weekly_aligned[i]
        curr_s3 = camarilla_s3_weekly_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price drops below weekly EMA34 (trend change)
            # 2. Price crosses below weekly Camarilla S3 (breakout failed)
            if (curr_close < curr_ema_34_1d or 
                curr_close < curr_s3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above weekly EMA34 (trend change)
            # 2. Price crosses above weekly Camarilla R3 (breakout failed)
            if (curr_close > curr_ema_34_1d or 
                curr_close > curr_r3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only enter on volume confirmation to avoid false breakouts
            if not curr_volume_confirm:
                signals[i] = 0.0
                continue
                
            # Long entry: price breaks above weekly Camarilla R3 + above weekly EMA34
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_34_1d):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below weekly Camarilla S3 + below weekly EMA34
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_34_1d):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals