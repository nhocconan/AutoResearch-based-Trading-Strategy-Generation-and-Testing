#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation
# Long when price breaks above R3 with 1d EMA34 uptrend and volume spike
# Short when price breaks below S3 with 1d EMA34 downtrend and volume spike
# Uses discrete position sizing (0.25) to minimize fee drag and targets 12-37 trades/year on 6h.
# Camarilla levels provide statistically significant support/resistance; breakouts with volume
# and trend alignment capture sustained moves while avoiding false breakouts in choppy markets.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    cam_high = df_1d['high'].values
    cam_low = df_1d['low'].values
    cam_close = df_1d['close'].values
    
    # Previous day's range
    prev_range = cam_high - cam_low
    r3 = cam_close + 1.1 * prev_range
    s3 = cam_close - 1.1 * prev_range
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # EMA34 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        trend_up = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] if i > 0 else False
        trend_down = ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] if i > 0 else False
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3, 1d EMA34 trending up, volume spike
            if price > r3_level and trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3, 1d EMA34 trending down, volume spike
            elif price < s3_level and trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on reversal or stoploss
            # Exit when price breaks below S3 (failed breakout) or EMA34 turns down
            if price < s3_level or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on reversal or stoploss
            # Exit when price breaks above R3 (failed breakdown) or EMA34 turns up
            if price > r3_level or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals