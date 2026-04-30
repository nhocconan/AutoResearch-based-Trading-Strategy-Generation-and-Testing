#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above R3, close > 1w EMA50, and volume > 2.0x 24-bar avg.
# Short when price breaks below S3, close < 1w EMA50, and volume > 2.0x 24-bar avg.
# Exit when price re-enters the Camarilla range (between S3 and R3).
# Uses 1d timeframe for lower trade frequency (target: 7-25 trades/year) to minimize fee drag.
# Camarilla levels provide clear breakout levels based on prior day's range.
# 1w EMA50 filters for higher timeframe trend alignment.
# Volume confirmation with moderate threshold reduces false breakouts.
# Works in bull markets via breakouts with trend and in bear markets via breakdowns with trend.
# Target: 30-100 total trades over 4 years.

name = "1d_Camarilla_R3_S3_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from prior 1d bar
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    # Using prior bar's OHLC to avoid look-ahead
    prior_close = np.roll(close, 1)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close[0] = np.nan  # First bar has no prior
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    camarilla_r3 = prior_close + 1.1 * (prior_high - prior_low) * 1.1 / 4
    camarilla_s3 = prior_close - 1.1 * (prior_high - prior_low) * 1.1 / 4
    
    # Volume confirmation: volume > 2.0x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, close > 1w EMA50, volume spike
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_50_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, close < 1w EMA50, volume spike
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_50_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price re-enters the Camarilla range (below R3)
            if curr_close < curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price re-enters the Camarilla range (above S3)
            if curr_close > curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals