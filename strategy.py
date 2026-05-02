#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses 4h primary timeframe for signal generation with Camarilla R3/S3 breakouts
# 1d EMA34 provides trend filter to avoid counter-trend trades (long only above EMA34, short only below)
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Discrete position sizing (0.25) minimizes fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Works in bull markets via trend-aligned breakouts, in bear via trend filter avoiding false signals
# Designed for moderate trade frequency to balance signal quality and fee drag

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    open_price = prices['open'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R2 = C + ((H-L)*1.1/6)
    #          S2 = C - ((H-L)*1.1/6), S1 = C - ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/2)
    # We need previous day's OHLC - resample to daily using actual Binance 1d data
    # Since we already have df_1d from get_htf_data, we can use it directly
    
    # For each 4h bar, we need the previous 1d bar's OHLC
    # We'll shift the 1d data by 1 bar to get previous day's values
    prev_day_open = df_1d['open'].values
    prev_day_high = df_1d['high'].values
    prev_day_low = df_1d['low'].values
    prev_day_close = df_1d['close'].values
    
    # Shift to get previous day's values (so today's 4h bars use yesterday's OHLC)
    prev_day_open_shifted = np.roll(prev_day_open, 1)
    prev_day_high_shifted = np.roll(prev_day_high, 1)
    prev_day_low_shifted = np.roll(prev_day_low, 1)
    prev_day_close_shifted = np.roll(prev_day_close, 1)
    # First value will be invalid (rolled from last), set to NaN
    prev_day_open_shifted[0] = np.nan
    prev_day_high_shifted[0] = np.nan
    prev_day_low_shifted[0] = np.nan
    prev_day_close_shifted[0] = np.nan
    
    # Align previous day's OHLC to 4h timeframe
    prev_day_open_aligned = align_htf_to_ltf(prices, df_1d, prev_day_open_shifted)
    prev_day_high_aligned = align_htf_to_ltf(prices, df_1d, prev_day_high_shifted)
    prev_day_low_aligned = align_htf_to_ltf(prices, df_1d, prev_day_low_shifted)
    prev_day_close_aligned = align_htf_to_ltf(prices, df_1d, prev_day_close_shifted)
    
    # Calculate Camarilla levels
    # R3 = C + ((H-L)*1.1/4)
    # S3 = C - ((H-L)*1.1/2)
    camarilla_r3 = prev_day_close_aligned + ((prev_day_high_aligned - prev_day_low_aligned) * 1.1 / 4)
    camarilla_s3 = prev_day_close_aligned - ((prev_day_high_aligned - prev_day_low_aligned) * 1.1 / 2)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators and previous day data)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla R3 + price > 1d EMA34 + volume confirm
            if close[i] > camarilla_r3[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 + price < 1d EMA34 + volume confirm
            elif close[i] < camarilla_s3[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla S3 or reverse signal
            if close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla R3 or reverse signal
            if close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals