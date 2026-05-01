#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout + 1d EMA34 trend + volume spike (>2x 20-bar MA)
# Camarilla levels provide precise intraday support/resistance, 1d EMA34 filters primary trend,
# volume spike confirms institutional participation. Works in bull via breakouts and bear via breakdowns.
# Target: 80-150 total trades over 4 years (20-38/year) with discrete sizing (0.30).

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # 1d HTF data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla levels from previous 1d bar
    # Need previous day's OHLC to calculate today's levels
    # We'll calculate daily OHLC from 4h data by resampling conceptually but using actual 1d data
    # Since we have df_1d, we can use its open, high, low, close
    # Camarilla R3, S3, R4, S4 levels
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    # But we need previous day's values, so we shift df_1d by 1
    
    # Calculate Camarilla levels for each 1d bar (using that day's OHLC)
    # Then align to 4h, but we need the levels from the PREVIOUS completed 1d bar
    # So we calculate levels on df_1d, then shift by 1 to get previous day's levels
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_open = df_1d['open'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels based on previous day
    rng = prev_high - prev_low
    camarilla_r3 = prev_close + (rng * 1.1 / 4)
    camarilla_s3 = prev_close - (rng * 1.1 / 4)
    camarilla_r4 = prev_close + (rng * 1.1 / 2)
    camarilla_s4 = prev_close - (rng * 1.1 / 2)
    
    # Align to 4h - these levels are valid for the entire 1d period following the previous day
    camarilla_r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_4h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_4h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20  # Need 20 for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_4h[i]) or np.isnan(camarilla_r3_4h[i]) or np.isnan(camarilla_s3_4h[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, above 1d EMA34, and volume spike
            if curr_close > camarilla_r3_4h[i] and curr_close > ema_34_4h[i] and vol_confirm:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Camarilla S3, below 1d EMA34, and volume spike
            elif curr_close < camarilla_s3_4h[i] and curr_close < ema_34_4h[i] and vol_confirm:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below Camarilla S3 or below 1d EMA34
            if curr_close < camarilla_s3_4h[i] or curr_close < ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit on price breaking above Camarilla R3 or above 1d EMA34
            if curr_close > camarilla_r3_4h[i] or curr_close > ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals