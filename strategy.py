#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike
# Long when price breaks above Camarilla R3 level in bullish regime (close > 12h EMA50) with volume > 2.0x 20-bar avg
# Short when price breaks below Camarilla S3 level in bearish regime (close < 12h EMA50) with volume spike
# Uses 12h EMA50 to filter for trending direction, avoiding counter-trend whipsaws
# Volume confirmation ensures breakouts have institutional participation
# Target: 20-35 trades/year (80-140 total over 4 years) to minimize fee drag
# Camarilla levels calculated from prior 12h bar's high-low-close (aligned to completed 12h bar)

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_Trend_v1"
timeframe = "4h"
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
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from prior completed 12h bar
    # R3 = close + 1.1*(high - low), S3 = close - 1.1*(high - low)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_r3 = close_12h + 1.1 * (high_12h - low_12h)
    camarilla_s3 = close_12h - 1.1 * (high_12h - low_12h)
    
    # Align Camarilla levels to 4h timeframe (completed 12h bar only)
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 50)  # warmup for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema_trend = ema_50_12h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend filter: bullish if close > 12h EMA50, bearish if close < 12h EMA50
        is_bullish = curr_close > curr_ema_trend
        is_bearish = curr_close < curr_ema_trend
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in correct trend direction
            if curr_volume_confirm:
                # Bullish breakout: price breaks above Camarilla R3 in bullish regime
                if is_bullish and curr_close > curr_r3:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below Camarilla S3 in bearish regime
                elif is_bearish and curr_close < curr_s3:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to Camarilla pivot (midpoint of H-L) OR breaks below S3 with volume
            camarilla_pivot = (curr_r3 + curr_s3) / 2.0  # approximate pivot
            
            if curr_close <= camarilla_pivot or (curr_close < curr_s3 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to Camarilla pivot OR breaks above R3 with volume
            camarilla_pivot = (curr_r3 + curr_s3) / 2.0  # approximate pivot
            
            if curr_close >= camarilla_pivot or (curr_close > curr_r3 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals