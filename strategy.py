#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation (>1.5x 20-period average)
# Camarilla pivot levels provide high-probability reversal/breakout zones; 1w EMA50 ensures alignment with weekly trend to avoid counter-trend whipsaws.
# Volume confirmation filters for institutional participation; discrete sizing (0.25) minimizes fee churn.
# Effective in both bull and bear markets: catches breakouts during strong trends, avoids false breakouts in chop.
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe.

name = "1d_Camarilla_R3S3_1wEMA50_VolumeConfirm_v1"
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
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels from previous day (using 1d OHLC from prices)
    # Camarilla levels: R3 = close + (high - low) * 1.1/2, S3 = close - (high - low) * 1.1/2
    # Use previous day's OHLC to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # first bar has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Calculate 20-period average volume for confirmation (on 1d timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 1)  # 1w EMA50 warmup (Camarilla uses prev bar, so start from 1)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Camarilla breakout conditions
        breakout_long = curr_high > curr_r3   # price breaks above R3
        breakout_short = curr_low < curr_s3   # price breaks below S3
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price retouches S3 OR trend turns bearish
            if curr_close < curr_s3 or curr_close < curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retouches R3 OR trend turns bullish
            if curr_close > curr_r3 or curr_close > curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bullish breakout above R3 AND above 1w EMA50 AND volume confirmation
            if (breakout_long and 
                curr_close > curr_ema_1w and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish breakout below S3 AND below 1w EMA50 AND volume confirmation
            elif (breakout_short and 
                  curr_close < curr_ema_1w and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals