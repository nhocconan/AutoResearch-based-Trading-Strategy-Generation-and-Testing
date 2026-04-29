#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (>1.5x 20-period average)
# Camarilla pivot levels (R3, S3) act as strong support/resistance where price often reverses or accelerates.
# Breakout above R3 or below S3 with volume indicates institutional participation and potential trend continuation.
# 1d EMA34 ensures we trade only with the higher timeframe trend to avoid whipsaws and counter-trend breakouts.
# Volume confirmation filters for meaningful participation; discrete sizing (0.25) minimizes fee churn.
# Effective in both bull and bear markets: captures strong momentum breakouts aligned with higher timeframe trend.
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
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
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    typical_price_1d_arr = typical_price_1d.values
    
    # Camarilla width = (high - low) * 1.1 / 12
    camarilla_width_1d = (df_1d['high'] - df_1d['low']) * 1.1 / 12.0
    camarilla_width_1d_arr = camarilla_width_1d.values
    
    # R3 = close + (high - low) * 1.1 * 1.1 / 4
    # S3 = close - (high - low) * 1.1 * 1.1 / 4
    camarilla_r3_1d = df_1d['close'] + camarilla_width_1d * 1.1 * 3.0
    camarilla_s3_1d = df_1d['close'] - camarilla_width_1d * 1.1 * 3.0
    camarilla_r3_1d_arr = camarilla_r3_1d.values
    camarilla_s3_1d_arr = camarilla_s3_1d.values
    
    # Align Camarilla levels to 4h timeframe (available after 1d bar closes)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d_arr)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d_arr)
    
    # Calculate 20-period average volume for confirmation (on 4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # 1d EMA34, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price breaks below S3 OR trend turns bearish (price below 1d EMA34)
            if curr_low < curr_s3 or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 OR trend turns bullish (price above 1d EMA34)
            if curr_high > curr_r3 or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND above 1d EMA34 AND volume confirmation
            if (curr_high > curr_r3 and 
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 AND below 1d EMA34 AND volume confirmation
            elif (curr_low < curr_s3 and 
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals