#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above R3 (from prior 12h) AND price > 12h EMA50 with volume spike
# Short when price breaks below S3 (from prior 12h) AND price < 12h EMA50 with volume spike
# Uses 12h EMA50 for trend filter to avoid counter-trend whipsaws
# Camarilla levels calculated from prior 12h bar (HLC of completed 12h bar)
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 12h calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe (completed 12h bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from prior 12h bar (HLC of completed 12h bar)
    # Typical price = (H + L + C) / 3
    typical_price_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3.0
    # Camarilla width = (H - L) * 1.1 / 12
    camarilla_width_12h = (df_12h['high'].values - df_12h['low'].values) * 1.1 / 12.0
    # R3 = CP + 3 * width, S3 = CP - 3 * width
    r3_12h = typical_price_12h + 3.0 * camarilla_width_12h
    s3_12h = typical_price_12h - 3.0 * camarilla_width_12h
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema50 = ema50_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish breakout: price breaks above R3 AND above 12h EMA50
                if curr_close > curr_r3 and curr_close > curr_ema50:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakdown: price breaks below S3 AND below 12h EMA50
                elif curr_close < curr_s3 and curr_close < curr_ema50:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price closes below 12h EMA50 OR price drops below R3 (failed breakout)
            if curr_close < curr_ema50 or curr_close < curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price closes above 12h EMA50 OR price rises above S3 (failed breakdown)
            if curr_close > curr_ema50 or curr_close > curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals