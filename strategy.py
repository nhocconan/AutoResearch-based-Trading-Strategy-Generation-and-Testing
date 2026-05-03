#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# Uses Williams Alligator (jaw/teeth/lips) to identify trendless markets and trigger
# entries only when all three lines are aligned (trending) + price breaks extreme.
# 1d EMA50 filters higher timeframe trend direction, volume spike confirms momentum.
# Designed for BTC/ETH to work in both bull and bear markets by only trading
# strong aligned trends, avoiding choppy markets where Alligator is sleeping.

name = "4h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Williams Alligator on 4h timeframe
    # Jaw: Blue line (13-period SMMA, shifted 8 bars)
    # Teeth: Red line (8-period SMMA, shifted 5 bars)
    # Lips: Green line (5-period SMMA, shifted 3 bars)
    def smma(src, length):
        # Smoothed Moving Average: similar to EMA but with alpha = 1/length
        if length < 1:
            return np.full_like(src, np.nan)
        result = np.full_like(src, np.nan, dtype=np.float64)
        alpha = 1.0 / length
        for i in range(len(src)):
            if np.isnan(src[i]):
                result[i] = np.nan
            elif i == 0:
                result[i] = src[i]
            else:
                if np.isnan(result[i-1]):
                    result[i] = src[i]
                else:
                    result[i] = result[i-1] + alpha * (src[i] - result[i-1])
        return result
    
    jaw = smma(high + low, 13)  # Using median price (H+L)/2 for Alligator
    teeth = smma(high + low, 8)
    lips = smma(high + low, 5)
    
    # Apply shifts (Alligator lines are shifted into the future)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted periods
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align to 4h timeframe (already on 4h, no HTF alignment needed for Alligator)
    # But we need to ensure we don't use future data - the shift already handles this
    
    # Volume regime: current 4h volume > 1.8x 30-period MA (moderate to reduce trades)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for SMMA
        # Get current values
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Alligator alignment: all three lines ordered (trending market)
        # For uptrend: Lips > Teeth > Jaw
        # For downtrend: Jaw > Teeth > Lips
        alligator_align_up = (lips_val > teeth_val) and (teeth_val > jaw_val)
        alligator_align_down = (jaw_val > teeth_val) and (teeth_val > lips_val)
        
        # Entry conditions
        # Long: Alligator aligned up AND price breaks above Lips with volume spike AND above 1d EMA50
        long_entry = alligator_align_up and (close[i] > lips_val) and vol_spike and (close[i] > ema_trend)
        # Short: Alligator aligned down AND price breaks below Jaw with volume spike AND below 1d EMA50
        short_entry = alligator_align_down and (close[i] < jaw_val) and vol_spike and (close[i] < ema_trend)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on Alligator alignment break or price crosses Teeth
            if not alligator_align_up or (close[i] < teeth_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on Alligator alignment break or price crosses Teeth
            if not alligator_align_down or (close[i] > teeth_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals