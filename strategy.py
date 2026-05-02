#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1d EMA34 trend + volume spike (2x avg)
# Uses 12h primary timeframe for Camarilla breakout signals
# 1d EMA34 confirms longer-term trend direction (avoids counter-trend trades)
# Volume confirmation (2.0x 20-period average) ensures strong participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Camarilla provides clear support/resistance, 1d EMA adds robust trend filter, volume confirms conviction
# Works in both bull and bear markets by only trading in direction of 1d trend

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34)
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Camarilla levels (based on previous day's OHLC)
    # Camarilla R3 = close + (high - low) * 1.1/4
    # Camarilla S3 = close - (high - low) * 1.1/4
    high_roll = pd.Series(high).rolling(window=2, min_periods=2).max()
    low_roll = pd.Series(low).rolling(window=2, min_periods=2).min()
    close_prev = pd.Series(close).shift(1)
    
    camarilla_r3 = close_prev + (high_roll - low_roll) * 1.1 / 4
    camarilla_s3 = close_prev - (high_roll - low_roll) * 1.1 / 4
    
    camarilla_r3_aligned = camarilla_r3.values
    camarilla_s3_aligned = camarilla_s3.values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Camarilla R3 breakout long: close > R3
            # Camarilla S3 breakout short: close < S3
            breakout_long = close[i] > camarilla_r3_aligned[i]
            breakout_short = close[i] < camarilla_s3_aligned[i]
            
            # 1d EMA34 trend filter: close > EMA for longs, close < EMA for shorts
            ema_long = close[i] > ema_34_1d_aligned[i]
            ema_short = close[i] < ema_34_1d_aligned[i]
            
            if breakout_long and ema_long and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif breakout_short and ema_short and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Camarilla S3 breakdown (close < S3) or trend reversal
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Camarilla R3 breakout (close > R3) or trend reversal
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals