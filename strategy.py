#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Volume-Weighted Average Price (VWAP) deviation with 12-hour trend filter
# Long when price deviates below VWAP by >1.5 sigma AND price > 12h EMA50 AND volume confirmation
# Short when price deviates above VWAP by >1.5 sigma AND price < 12h EMA50 AND volume confirmation
# Exit when price returns to VWAP (±0.5 sigma)
# Uses mean reversion to VWAP in ranging markets, filtered by 12h trend to avoid counter-trend trades
# Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate VWAP and standard deviation (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vp = typical_price * volume
    cum_vp = np.nancumsum(vp)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_vp, cum_vol, out=np.full_like(cum_vp, np.nan), where=cum_vol!=0)
    
    # Calculate VWAP deviation standard deviation (using 20-period)
    vwap_dev = typical_price - vwap
    vwap_ma = pd.Series(vwap_dev).rolling(window=20, min_periods=20).mean().values
    vwap_std = pd.Series(vwap_dev).rolling(window=20, min_periods=20).std().values
    vwap_z = np.divide((vwap_dev - vwap_ma), vwap_std, out=np.full_like(vwap_dev, np.nan), where=vwap_std!=0)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap[i]) or np.isnan(vwap_z[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap_val = vwap[i]
        z_score = vwap_z[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price below VWAP by >1.5 sigma AND above 12h EMA50 AND volume confirmation
            if (z_score < -1.5 and price > ema50_12h_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price above VWAP by >1.5 sigma AND below 12h EMA50 AND volume confirmation
            elif (z_score > 1.5 and price < ema50_12h_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP (within +/- 0.5 sigma)
            if abs(z_score) < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to VWAP (within +/- 0.5 sigma)
            if abs(z_score) < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_VWAP_MeanReversion_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0