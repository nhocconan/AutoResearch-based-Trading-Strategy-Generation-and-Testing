#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla levels provide intraday support/resistance; breakout above R3 or below S3 indicates strong momentum
# Combined with 1d EMA34 trend filter ensures alignment with higher timeframe direction
# Volume spike confirms institutional participation
# Target: 20-50 trades/year on 4h to minimize fee drag while capturing strong moves
# Works in bull markets via breakout longs and in bear markets via breakdown shorts aligned with daily trend

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # 1d HTF data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for 4h timeframe using previous day's OHLC
    # We need to get daily OHLC from 1d data and align to 4h bars
    # For each 4h bar, we use the previous completed 1d bar's OHLC
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get previous day's OHLC (shifted by 1 to avoid look-ahead)
    prev_close_1d = np.roll(df_1d['close'].values, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_open_1d = np.roll(df_1d['open'].values, 1)
    # Set first value to NaN since there's no previous day
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_open_1d[0] = np.nan
    
    # Align previous day's OHLC to 4h timeframe
    prev_close_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    prev_open_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_open_1d)
    
    # Calculate Camarilla levels
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    rng = prev_high_1d_aligned - prev_low_1d_aligned
    r3 = prev_close_1d_aligned + rng * (1.1 / 4)
    s3 = prev_close_1d_aligned - rng * (1.1 / 4)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 20)  # Need sufficient history for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(prev_close_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, volume spike, uptrend
            if close[i] > r3[i] and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, volume spike, downtrend
            elif close[i] < s3[i] and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price below R3 or trend reversal
            if close[i] < r3[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price above S3 or trend reversal
            if close[i] > s3[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals