#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (>1.8x 20-period average)
# Camarilla pivot levels provide institutional support/resistance; R3/S3 are strong breakout levels.
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws.
# Volume confirmation filters for institutional participation; discrete sizing (0.25) minimizes fee churn.
# Effective in both bull and bear markets: catches breakouts during strong trends, avoids false breakouts in chop.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

name = "12h_Camarilla_R3S3_1dEMA34_VolumeConfirm_v1"
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
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Typical price = (high + low + close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_values = typical_price.values
    
    # Camarilla width = (high - low) * 1.1 / 12
    camarilla_width = ((df_1d['high'] - df_1d['low']) * 1.1 / 12).values
    
    # R3 = close + (high - low) * 1.1 / 4
    # S3 = close - (high - low) * 1.1 / 4
    camarilla_r3 = (typical_price_values + camarilla_width * 3).values  # 3 * width = 1.1/4 * (high-low)
    camarilla_s3 = (typical_price_values - camarilla_width * 3).values
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 20-period average volume for confirmation (on 12h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # 1d EMA34, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = curr_volume > 1.8 * curr_vol_ma
        
        # Camarilla breakout conditions
        breakout_long = curr_high > curr_r3   # price breaks above R3
        breakout_short = curr_low < curr_s3   # price breaks below S3
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price retouches S3 level OR trend turns bearish
            if curr_close < curr_s3 or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retouches R3 level OR trend turns bullish
            if curr_close > curr_r3 or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bullish breakout AND above 1d EMA34 AND volume confirmation
            if (breakout_long and 
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish breakout AND below 1d EMA34 AND volume confirmation
            elif (breakout_short and 
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals