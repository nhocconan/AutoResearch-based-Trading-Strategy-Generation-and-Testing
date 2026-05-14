#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with volume spike and 1w EMA50 trend filter
# Uses wider Camarilla levels (R4/S4 = C ± (H-L)*1.1/2) for stronger breakout signals
# Volume spike (>2.0x 30-period average) confirms institutional participation
# 1w EMA50 trend filter ensures trades align with weekly timeframe momentum
# Works in bull/bear: volume confirms breakout validity, 1w EMA50 filters counter-trend noise
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Camarilla_R4S4_VolumeSpike_1wEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R4/S4 = C ± (H-L)*1.1/2 (wider breakout bands)
    camarilla_range = (prev_high - prev_low) * 1.1 / 2.0
    r4 = prev_close + camarilla_range
    s4 = prev_close - camarilla_range
    
    # Align daily levels to 4h timeframe (wait for daily bar to close)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_ma_30)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(30, 50)  # warmup for volume MA and 1w EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma_30[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50 = ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: price breaks above R4 with volume and above 1w EMA50
                if curr_high > curr_r4 and curr_close > curr_ema_50:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below S4 with volume and below 1w EMA50
                elif curr_low < curr_s4 and curr_close < curr_ema_50:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price breaks below S4 (reversal signal)
            if curr_low < curr_s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit when price breaks above R4 (reversal signal)
            if curr_high > curr_r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals