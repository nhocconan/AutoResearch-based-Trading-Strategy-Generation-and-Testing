#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
# Enter long when price breaks above Camarilla R3, 12h EMA50 trending up, and volume > 2.0x 24-bar average.
# Enter short when price breaks below Camarilla S3, 12h EMA50 trending down, and volume > 2.0x 24-bar average.
# Exit when price returns to Camarilla pivot point (PP).
# Uses discrete position sizing (0.25) to limit drawdown. Target: 60-120 total trades over 4 years (15-30/year).
# Camarilla levels provide precise intraday support/resistance; 12h EMA50 ensures alignment with higher timeframe trend;
# volume confirmation filters weak breakouts. Works in both bull (breakouts) and bear (breakdowns).

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Camarilla levels from prior 12h bar (HLC of previous completed 12h bar)
    # We need to use the prior completed 12h bar's H, L, C to calculate levels for current 6h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each completed 12h bar
    # R4 = C + ((H-L)*1.1/2)
    # R3 = C + ((H-L)*1.1/4)
    # PP = (H+L+C)/3
    # S3 = C - ((H-L)*1.1/4)
    # S4 = C - ((H-L)*1.1/2)
    camarilla_r3 = close_12h + ((high_12h - low_12h) * 1.1 / 4)
    camarilla_s3 = close_12h - ((high_12h - low_12h) * 1.1 / 4)
    camarilla_pp = (high_12h + low_12h + close_12h) / 3
    
    # Align Camarilla levels to 6h (these levels are valid for the entire 6h bar following the 12h bar close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pp)
    
    # Volume confirmation: >2.0x 24-bar average volume (4 * 6h = 24h ≈ 1d)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 12h EMA50 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_50_aligned[i] - ema_50_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        # Price action
        price = close[i]
        
        # Breakout conditions
        breakout_long = price > camarilla_r3_aligned[i]
        breakout_short = price < camarilla_s3_aligned[i]
        
        # Exit conditions: return to pivot point
        exit_long = price < camarilla_pp_aligned[i]
        exit_short = price > camarilla_pp_aligned[i]
        
        # Handle entries and exits
        if breakout_long and ema_trend_up and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_short and ema_trend_down and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals