#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1H4D_Squeeze_Breakout_Volume - Bollinger Band squeeze on daily timeframe indicates low volatility,
# followed by breakout on 1H with volume confirmation. Works in both bull/bear markets as volatility compression
# precedes expansion regardless of direction. Uses 4H trend filter to avoid counter-trend entries.
# Target: 15-35 trades/year by requiring squeeze condition + breakout + volume + trend alignment.
name = "1H4D_Squeeze_Breakout_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Bollinger Bands and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Bollinger Bands (20, 2.0)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Bollinger Band squeeze: width below 20-period mean width
    mean_width = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < mean_width
    
    # 4H trend filter: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Align daily indicators to 1H
    squeeze_1h = align_htf_to_ltf(prices, df_1d, squeeze)
    ema_50_4h_1h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(40, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(squeeze_1h[i]) or np.isnan(ema_50_4h_1h[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        is_squeeze = squeeze_1h[i]
        trend_4h = ema_50_4h_1h[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Enter long: break above upper BB with volume and above 4H trend
            if close[i] > upper_bb[i//24] and close[i] > trend_4h and vol_ok and is_squeeze:
                signals[i] = 0.20
                position = 1
            # Enter short: break below lower BB with volume and below 4H trend
            elif close[i] < lower_bb[i//24] and close[i] < trend_4h and vol_ok and is_squeeze:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: close below middle BB or trend breaks
            if close[i] < sma_20[i//24] or close[i] < trend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: close above middle BB or trend breaks
            if close[i] > sma_20[i//24] or close[i] > trend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals