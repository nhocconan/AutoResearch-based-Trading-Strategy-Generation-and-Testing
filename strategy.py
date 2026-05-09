#!/usr/bin/env python3
# Hypothesis: 1h ATR breakout with 4h trend filter and volume spike for BTC/ETH
# Long when price breaks above ATR-based upper band with 4h EMA50 uptrend and volume > 1.8x average
# Short when price breaks below ATR-based lower band with 4h EMA50 downtrend and volume > 1.8x average
# Exit when price retouches the middle SMA or reverses to opposite ATR band
# Uses ATR for volatility-adjusted breakouts, EMA for trend, volume for conviction
# Target: 60-150 total trades over 4 years (15-37/year) with size 0.20
# 4h EMA50 trend filter reduces whipsaw in ranging markets and captures trends

name = "1h_ATR_Breakout_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 14-period ATR (volatility measure)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period SMA (middle band)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR-based bands (multiplier 2.0)
    upper_band = sma20 + (2.0 * atr)
    lower_band = sma20 - (2.0 * atr)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(sma20[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper band, 4h EMA50 uptrend, volume spike
            if (close[i] > upper_band[i] and 
                ema50_4h_aligned[i] > ema50_4h_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below lower band, 4h EMA50 downtrend, volume spike
            elif (close[i] < lower_band[i] and 
                  ema50_4h_aligned[i] < ema50_4h_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price retouches middle SMA or reverses to lower band
            if (close[i] <= sma20[i]) or (close[i] < lower_band[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price retouches middle SMA or reverses to upper band
            if (close[i] >= sma20[i]) or (close[i] > upper_band[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals