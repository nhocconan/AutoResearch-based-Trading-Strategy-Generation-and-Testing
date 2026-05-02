#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 Breakout + 1d Volume Spike + 1d EMA50 Trend
# Camarilla pivot levels (R3/S3) act as intraday support/resistance. A break above R3 or below S3
# with volume confirmation and alignment to the daily EMA50 trend captures momentum moves.
# Works in bull/bear markets by filtering breaks with higher-timeframe trend and volume.
# Target: 50-150 trades over 4 years (12-37/year) on 6h.

name = "6h_Camarilla_R3S3_Breakout_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume average for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Camarilla pivots from previous 1d bar (OHLC)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We use previous completed 1d bar to avoid look-ahead
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan  # First bar has no previous
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3_1d = prev_close_1d + ((prev_high_1d - prev_low_1d) * 1.1 / 4)
    camarilla_s3_1d = prev_close_1d - ((prev_high_1d - prev_low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for 1d EMA50 and volume)
    start_idx = 50  # 1d EMA50 warmup
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_r3_1d_aligned[i]) or 
            np.isnan(camarilla_s3_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 with volume spike AND price > 1d EMA50 (bullish trend)
            if (close[i] > camarilla_r3_1d_aligned[i] and 
                volume[i] > (2.0 * vol_ma_1d_aligned[i]) and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S3 with volume spike AND price < 1d EMA50 (bearish trend)
            elif (close[i] < camarilla_s3_1d_aligned[i] and 
                  volume[i] > (2.0 * vol_ma_1d_aligned[i]) and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Camarilla S3 (mean reversion) OR price below 1d EMA50 (trend change)
            if close[i] < camarilla_s3_1d_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla R3 (mean reversion) OR price above 1d EMA50 (trend change)
            if close[i] > camarilla_r3_1d_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals