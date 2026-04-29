#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation (>1.5x 24-period average)
# Uses 4h for signal direction and 1h for precise entry timing. Camarilla levels identify key support/resistance
# where institutional order flow concentrates. Breakouts above R3 or below S3 with volume confirmation indicate
# strong momentum. 4h EMA50 ensures we trade with the higher timeframe trend to avoid whipsaws in choppy markets.
# Volume filter ensures participation. Discrete sizing (0.20) minimizes fee churn. Session filter (08-20 UTC) reduces noise.
# Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 levels
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 24-period average volume for confirmation (on 1h timeframe)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # 4h EMA50, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_4h = ema_50_4h_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_vol_ma = vol_ma_24[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 24-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Breakout conditions
        # Long breakout: price breaks above R3 level
        # Short breakout: price breaks below S3 level
        breakout_long = curr_high > curr_r3
        breakout_short = curr_low < curr_s3
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below 4h EMA50 OR opposite breakout occurs
            if curr_close < curr_ema_4h or breakout_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h EMA50 OR opposite breakout occurs
            if curr_close > curr_ema_4h or breakout_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: bullish breakout AND above 4h EMA50 AND volume confirmation
            if (breakout_long and 
                curr_close > curr_ema_4h and 
                vol_confirm):
                signals[i] = 0.20
                position = 1
            # Short entry: bearish breakout AND below 4h EMA50 AND volume confirmation
            elif (breakout_short and 
                  curr_close < curr_ema_4h and 
                  vol_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals