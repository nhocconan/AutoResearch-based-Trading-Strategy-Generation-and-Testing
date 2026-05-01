#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H4/L4 breakout with 1d EMA50 trend filter and volume spike confirmation.
# Uses Camarilla H4/L4 levels for mean reversion in ranging markets, filtered by 1d EMA50 trend and volume > 2x 20-period median.
# Works in bull (sell at H4 in uptrend, buy at L4 in downtrend) and bear (buy at L4 in downtrend, sell at H4 in uptrend).
# Discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.

name = "12h_Camarilla_H4L4_MeanReversion_1dEMA50_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Camarilla levels from previous day OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla H4 and L4 levels (mean reversion zones)
    camarilla_h4 = prev_close + (prev_high - prev_low) * 1.125
    camarilla_l4 = prev_close - (prev_high - prev_low) * 1.125
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA50 and volume median
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        # Camarilla mean reversion conditions (H4/L4 for reversal)
        touch_h4 = curr_close >= camarilla_h4_aligned[i]   # touch or break above H4
        touch_l4 = curr_close <= camarilla_l4_aligned[i]   # touch or break below L4
        
        if position == 0:  # Flat - look for new entries
            # Short: Touch H4 AND uptrend AND volume confirmation (mean reversion short)
            if touch_h4 and uptrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            # Long: Touch L4 AND downtrend AND volume confirmation (mean reversion long)
            elif touch_l4 and downtrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on touch H4 (mean reversion target)
            if touch_h4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on touch L4 (mean reversion target)
            if touch_l4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals