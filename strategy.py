#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation
# Camarilla pivot levels provide high-probability intraday support/resistance, especially R3/S3 levels
# which often act as breakout/breakdown zones. Combined with 1d EMA50 for trend alignment and
# volume spike for institutional confirmation, this strategy aims to capture strong momentum moves
# while filtering false breakouts. Designed to work in both bull and bear markets by trading
# with the higher-timeframe trend. Target: 25-40 trades/year (100-160 over 4 years) to minimize fee drag.

name = "4h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical Camarilla formula based on previous day's OHLC
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # We'll calculate daily pivot levels and align to 4h timeframe
    df_1d['typical_price'] = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    df_1d['range'] = df_1d['high'] - df_1d['low']
    camarilla_r3 = df_1d['close'] + (df_1d['range'] * 1.1 / 2)
    camarilla_s3 = df_1d['close'] - (df_1d['range'] * 1.1 / 2)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20)  # Need sufficient history for 1d EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_r3_aligned[i]  # Price breaks above R3
        breakout_down = close[i] < camarilla_s3_aligned[i]  # Price breaks below S3
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla R3 breakout up, volume spike, uptrend
            if breakout_up and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla S3 breakout down, volume spike, downtrend
            elif breakout_down and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price re-entering Camarilla range or trend reversal
            if close[i] < camarilla_r3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price re-entering Camarilla range or trend reversal
            if close[i] > camarilla_s3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals