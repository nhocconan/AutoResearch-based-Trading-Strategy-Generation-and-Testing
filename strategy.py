#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels provide precise support/resistance based on prior day's range
# Breakout of R3 (resistance 3) or S3 (support 3) with volume confirmation (>2x 20-period average) 
# indicates strong momentum. 1d EMA34 filter ensures trades align with daily trend.
# Designed for ~20-40 trades/year on 4h timeframe to minimize fee drag and avoid overtrading.
# Uses discrete position sizing (0.25) to reduce churn and manage drawdown.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Get 1d data for EMA34 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from prior 1d bar (using get_htf_data with shift)
    # We need prior day's OHLC, so we use the 1d data shifted by 1 to avoid look-ahead
    df_1d_shift = df_1d.copy()
    # Calculate typical Camarilla levels: based on prior day's range
    # R4 = Close + 1.5*(High-Low), R3 = Close + 1.1*(High-Low), etc.
    # But we use the prior completed 1d bar, so we shift the 1d data by 1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift to get prior day's OHLC (avoid look-ahead)
    high_1d_prev = np.concatenate([[np.nan], high_1d[:-1]])
    low_1d_prev = np.concatenate([[np.nan], low_1d[:-1]])
    close_1d_prev = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Calculate Camarilla levels for prior day
    rang = high_1d_prev - low_1d_prev
    R3 = close_1d_prev + 1.1 * rang
    S3 = close_1d_prev - 1.1 * rang
    
    # Align these levels to 4h timeframe (they update only when new 1d bar completes)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Calculate 20-period average volume for confirmation (on 4h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_R3 = R3_aligned[i]
        curr_S3 = S3_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits: reverse position if opposite signal triggers, or flatten on trend reversal
        if position == 1:  # Long position
            # Exit: price breaks below S3 (support 3) or trend turns bearish
            if curr_low < curr_S3 or curr_close < curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 (resistance 3) or trend turns bullish
            if curr_high > curr_R3 or curr_close > curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average (strict to reduce trades)
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: price breaks above R3 with volume confirmation and bullish trend
            if vol_confirm and curr_high > curr_R3 and curr_close > curr_ema34_1d:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume confirmation and bearish trend
            elif vol_confirm and curr_low < curr_S3 and curr_close < curr_ema34_1d:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals