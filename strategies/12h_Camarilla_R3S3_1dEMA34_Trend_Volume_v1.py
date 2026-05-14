#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses 1d EMA34 for structural trend bias (long when price > EMA34, short when price < EMA34)
# Camarilla R3/S3 levels provide precise entry/exit levels derived from 1d range
# Volume confirmation > 1.8x 20-period EMA ensures institutional participation
# Designed for low trade frequency: ~12-37 trades/year per symbol with 0.28 sizing
# 1d EMA34 filter reduces false breakouts in choppy markets while capturing strong trends
# Works in both bull and bear markets by following the dominant daily trend

name = "12h_Camarilla_R3S3_1dEMA34_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla R3 = close + 1.1*(high-low)*1.1/4
    # Camarilla S3 = close - 1.1*(high-low)*1.1/4
    typical_range = df_1d['high'] - df_1d['low']
    camarilla_r3 = df_1d['close'] + 1.1 * typical_range * 1.1 / 4
    camarilla_s3 = df_1d['close'] - 1.1 * typical_range * 1.1 / 4
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Volume confirmation: volume > 1.8 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1d data for EMA34 (35 days) + Camarilla needs 1d
    start_idx = 35
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34: long above EMA34, short below EMA34
        bullish_bias = close[i] > ema_34_1d_aligned[i]
        bearish_bias = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: Camarilla R3 breakout above with volume spike
                if close[i] > camarilla_r3_aligned[i] and volume_spike[i]:
                    signals[i] = 0.28
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: Camarilla S3 breakdown below with volume spike
                if close[i] < camarilla_s3_aligned[i] and volume_spike[i]:
                    signals[i] = -0.28
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA34
        
        elif position == 1:  # Long position
            # Exit: Camarilla S3 breakdown below (failure of breakout)
            if close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:  # Short position
            # Exit: Camarilla R3 breakout above (failure of breakdown)
            if close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals