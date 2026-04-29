#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with volume confirmation and 1w EMA50 trend filter
# Uses proven Camarilla pivot structure (R4/S4 = stronger reversal levels) with 1w EMA50 trend and volume spike confirmation
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Works in bull/bear: volume spike confirms institutional interest, 1w EMA50 filters counter-trend noise
# Novelty: R4/S4 levels (H-L)*1.1 (wider bands) reduce false breakouts vs R3/S3, improving trade quality

name = "12h_Camarilla_R4S4_VolumeSpike_1wEMA50_Trend_v1"
timeframe = "12h"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R4/S4 = C ± (H-L)*1.1 (wider bands than R3/S3)
    camarilla_range = (prev_high - prev_low) * 1.1
    r4 = prev_close + camarilla_range
    s4 = prev_close - camarilla_range
    
    # Align daily levels to 12h timeframe (wait for daily bar to close)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50)  # warmup for volume MA and 1w EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_50_aligned[i]):
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
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below S4 with volume and below 1w EMA50
                elif curr_low < curr_s4 and curr_close < curr_ema_50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price breaks below S4 (reversal signal)
            if curr_low < curr_s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price breaks above R4 (reversal signal)
            if curr_high > curr_r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals