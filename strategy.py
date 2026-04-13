#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h primary with 1d HTF - Camarilla pivot breakout with volume confirmation
    # Camarilla R4/S4 levels act as strong intraday support/resistance from institutional order flow
    # Breakouts beyond R4/S4 with volume confirmation capture genuine institutional moves
    # Works in both bull/bear markets by trading breakouts in direction of 1d trend
    # Target: 50-150 total trades over 4 years (12-37/year) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We use previous day's range to calculate today's levels
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Avoid division by zero and handle first bar
    prev_high_1d[0] = prev_high_1d[1] if len(prev_high_1d) > 1 else prev_high_1d[0]
    prev_low_1d[0] = prev_low_1d[1] if len(prev_low_1d) > 1 else prev_low_1d[0]
    prev_close_1d[0] = prev_close_1d[1] if len(prev_close_1d) > 1 else prev_close_1d[0]
    
    range_1d = prev_high_1d - prev_low_1d
    camarilla_r4 = prev_close_1d + range_1d * 1.1 / 2
    camarilla_s4 = prev_close_1d - range_1d * 1.1 / 2
    camarilla_r3 = prev_close_1d + range_1d * 1.1 / 4
    camarilla_s3 = prev_close_1d - range_1d * 1.1 / 4
    
    # 1d trend filter: EMA(21) - only trade breakouts in direction of daily trend
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 6h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)  # volume uses 1d alignment for daily average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_21_1d_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * vol_avg_20_aligned[i]
        
        # Breakout conditions beyond R4/S4 (strong institutional levels)
        breakout_up = close[i] > camarilla_r4_aligned[i]
        breakout_down = low[i] < camarilla_s4_aligned[i]  # use low for downside breakout
        
        # Trend filter: only trade breakouts in direction of 1d EMA
        trend_up = close[i] > ema_21_1d_aligned[i]
        trend_down = close[i] < ema_21_1d_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and volume_confirmed and trend_up
        enter_short = breakout_down and volume_confirmed and trend_down
        
        # Exit conditions: price returns to Camarilla R3/S3 levels (profit taking)
        exit_long = position == 1 and close[i] < camarilla_r3_aligned[i]
        exit_short = position == -1 and close[i] > camarilla_s3_aligned[i]
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0