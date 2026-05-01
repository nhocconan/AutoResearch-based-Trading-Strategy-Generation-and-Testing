#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when: price breaks above Camarilla R3 level AND price > 1d EMA34 AND volume > 2.0x 20-bar average
# Short when: price breaks below Camarilla S3 level AND price < 1d EMA34 AND volume > 2.0x 20-bar average
# Exit when: price crosses 10-period EMA (dynamic stop) OR opposite Camarilla breakout occurs
# Uses Camarilla pivots for structure (proven edge), 1d EMA for trend alignment, volume for conviction.
# Target: 20-40 trades/year on 4h. Discrete sizing 0.25 to minimize fee drag while capturing strong trends.
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) by trading with aligned 1d trend.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeBreakout_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for Camarilla calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # For intraday, we use previous 4h bar's high/low/close
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    prev_close_4h = df_4h['close'].shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 4
    camarilla_s3 = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 4
    
    # Align Camarilla levels to 4h primary timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate 4h 10-period EMA for dynamic exit
    close_4h = df_4h['close'].values
    ema_10_4h = pd.Series(close_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_10_4h)
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h primary timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for EMA10 (10) + EMA34 (34) + Camarilla (need prev bar)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_10_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_ema_10 = ema_10_4h_aligned[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        # Volume confirmation: current 4h volume > 2.0x 20-period average
        vol_4h = df_4h['volume'].values
        vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
        vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
        curr_vol_ma = vol_ma_4h_aligned[i]
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Camarilla R3 + price > 1d EMA34 + volume confirmation
            if (curr_high > curr_r3 and 
                curr_close > curr_ema_34_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S3 + price < 1d EMA34 + volume confirmation
            elif (curr_low < curr_s3 and 
                  curr_close < curr_ema_34_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below 10-period EMA OR opposite Camarilla breakout
            if (curr_close < curr_ema_10) or \
               (curr_low < curr_s3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above 10-period EMA OR opposite Camarilla breakout
            if (curr_close > curr_ema_10) or \
               (curr_high > curr_r3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals