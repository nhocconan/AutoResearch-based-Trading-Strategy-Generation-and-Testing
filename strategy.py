#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4-hour Camarilla R3/S3 breakout with 12-hour EMA34 trend filter and volume confirmation
    # Camarilla levels provide high-probability reversal/continuation points based on institutional order flow.
    # Breakouts at R3/S3 with volume confirmation and higher timeframe trend alignment yield high win rates.
    # Designed to work in both bull and bear markets by following 12h trend and requiring volume confirmation.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla Pivots (based on previous 4h bar)
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # We'll use R3 and S3 as our entry/exit levels
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    # First value will be incorrect due to roll, but will be handled by min_periods equivalent
    cam_R3_4h = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 4
    cam_S3_4h = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    cam_R3_aligned = align_htf_to_ltf(prices, df_4h, cam_R3_4h)
    cam_S3_aligned = align_htf_to_ltf(prices, df_4h, cam_S3_4h)
    
    # Load 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(cam_R3_aligned[i]) or np.isnan(cam_S3_aligned[i]) or
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above R3 with volume + price above 12h EMA34 (uptrend)
            if close[i] > cam_R3_aligned[i] and vol_spike[i] and close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S3 with volume + price below 12h EMA34 (downtrend)
            elif close[i] < cam_S3_aligned[i] and vol_spike[i] and close[i] < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Camarilla level (S3 for long, R3 for short) or trend reversal
            if position == 1:
                if close[i] < cam_S3_aligned[i] or close[i] < ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > cam_R3_aligned[i] or close[i] > ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hEMA34_Volume_Session_v1"
timeframe = "4h"
leverage = 1.0