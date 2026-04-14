#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation
# Camarilla levels (R3/R4, S3/S4) act as strong support/resistance in both bull and bear markets
# Breakout above R4 or below S4 with volume > 1.5x 12h average signals continuation
# Fade at R3/S3 with volume > 2x average for mean reversion in ranging markets
# Only trade when 12h ADX > 25 to avoid choppy markets
# Target: 15-30 trades/year per symbol (60-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for Camarilla and ADX
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla levels from previous 12h bar
    # Typical price = (H + L + C) / 3
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    range_hl = df_12h['high'] - df_12h['low']
    
    # Camarilla levels
    r3 = typical_price + 1.1 * range_hl / 2
    r4 = typical_price + 1.1 * range_hl
    s3 = typical_price - 1.1 * range_hl / 2
    s4 = typical_price - 1.1 * range_hl
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3.values)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4.values)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4.values)
    
    # Calculate 12h ADX for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume average (20 periods on 12h)
    vol_ma_12h = pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, 20, 34)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Breakout continuation: price breaks R4/S4 with volume confirmation
            volume_confirmed = volume[i] > 1.5 * vol_ma_aligned[i]
            
            if trending and volume_confirmed:
                # Long breakout above R4
                if close[i] > r4_aligned[i-1]:
                    position = 1
                    signals[i] = position_size
                # Short breakdown below S4
                elif close[i] < s4_aligned[i-1]:
                    position = -1
                    signals[i] = -position_size
            # Mean reversion fade at R3/S3 in ranging markets
            elif not trending:
                volume_extreme = volume[i] > 2.0 * vol_ma_aligned[i]
                if volume_extreme:
                    # Fade at R3 (sell pressure)
                    if close[i] > r3_aligned[i-1]:
                        position = -1
                        signals[i] = -position_size
                    # Fade at S3 (buying pressure)
                    elif close[i] < s3_aligned[i-1]:
                        position = 1
                        signals[i] = position_size
        elif position == 1:
            # Exit long: price returns to S3 or breaks below S4
            if close[i] < s3_aligned[i] or close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to R3 or breaks above R4
            if close[i] > r3_aligned[i] or close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_Camarilla_Breakout_Volume_ADX_v1"
timeframe = "6h"
leverage = 1.0