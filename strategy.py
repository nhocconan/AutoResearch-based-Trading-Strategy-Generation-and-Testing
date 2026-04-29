#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H3/L3 breakout with volume spike and 1d EMA50 trend filter
# Long when price breaks above H3 with volume confirmation and price > 1d EMA50
# Short when price breaks below L3 with volume confirmation and price < 1d EMA50
# Uses tighter H3/L3 levels (vs R3/S3) for higher quality signals
# Volume confirmation filters low-quality breakouts
# 1d EMA50 ensures trading with higher timeframe trend
# Target: 75-150 total trades over 4 years (19-38/year) for optimal fee drag balance

name = "4h_Camarilla_H3L3_VolumeSpike_1dEMA50_v1"
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
    
    # Calculate Camarilla pivot levels from previous day
    # Need to get daily data first
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Camarilla levels: H3/L3 = C ± (H-L)*1.1/4
    camarilla_range = (prev_high - prev_low) * 1.1 / 4
    h3 = prev_close + camarilla_range
    l3 = prev_close - camarilla_range
    
    # Align daily levels to 4h timeframe (wait for daily bar to close)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50)  # warmup for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_h3 = h3_aligned[i]
        curr_l3 = l3_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50 = ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: price breaks above H3 with volume and above 1d EMA50
                if curr_high > curr_h3 and curr_close > curr_ema_50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below L3 with volume and below 1d EMA50
                elif curr_low < curr_l3 and curr_close < curr_ema_50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price breaks below L3 (reversal signal)
            if curr_low < curr_l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price breaks above H3 (reversal signal)
            if curr_high > curr_h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals