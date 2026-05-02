#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R3/S3 Breakout + 1d EMA34 Trend + Volume Spike Confirmation
# Camarilla pivot levels provide intraday support/resistance with high probability reversal/breakout points
# R3 (resistance 3) and S3 (support 3) are strong levels where breaks often lead to sustained moves
# Only trade breakouts in direction of 1d EMA34 trend to avoid counter-trend whipsaws
# Volume spike (2.0x 20-period avg) confirms institutional participation
# Discrete position sizing (0.25) minimizes fee churn
# Targets 20-50 trades/year (80-200 total over 4 years) for 4h timeframe
# Works in bull markets via buying breakouts in uptrend and in bear markets via selling breakdowns in downtrend

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical Camarilla formula based on previous day's OHLC
    # We need to shift by 1 to use previous day's data
    if hasattr(high, 'reshape'):
        # For safety, ensure we're working with 1D arrays
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
    else:
        high_series = high
        low_series = low
        close_series = close
    
    # Calculate daily OHLC (we'll use the HTF data for accuracy)
    # Since we're on 4h timeframe, we need to get daily data properly
    # Use the 1d data we already loaded
    if len(df_1d) >= 2:
        prev_day_high = df_1d['high'].shift(1).values
        prev_day_low = df_1d['low'].shift(1).values
        prev_day_close = df_1d['close'].shift(1).values
        
        # Align these to 4h timeframe
        prev_day_high_aligned = align_htf_to_ltf(prices, df_1d, prev_day_high)
        prev_day_low_aligned = align_htf_to_ltf(prices, df_1d, prev_day_low)
        prev_day_close_aligned = align_htf_to_ltf(prices, df_1d, prev_day_close)
        
        # Calculate Camarilla levels
        # Camarilla: Range = (High - Low)
        # R4 = Close + Range * 1.1/2
        # R3 = Close + Range * 1.1/4
        # S3 = Close - Range * 1.1/4
        # S4 = Close - Range * 1.1/2
        daily_range = prev_day_high_aligned - prev_day_low_aligned
        r3 = prev_day_close_aligned + daily_range * 1.1 / 4
        s3 = prev_day_close_aligned - daily_range * 1.1 / 4
    else:
        # Fallback: use rolling window if HTF data insufficient
        daily_high = pd.Series(high).rolling(window=6, min_periods=6).max().shift(6).values  # approx 1d from 6x4h
        daily_low = pd.Series(low).rolling(window=6, min_periods=6).min().shift(6).values
        daily_close = pd.Series(close).rolling(window=6, min_periods=6).mean().shift(6).values
        daily_range = daily_high - daily_low
        r3 = daily_close + daily_range * 1.1 / 4
        s3 = daily_close - daily_range * 1.1 / 4
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla calculation and volume MA)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above R3 + 1d close > EMA34 (uptrend) + volume spike
            if (close[i] > r3[i] and 
                close[i] > ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 + 1d close < EMA34 (downtrend) + volume spike
            elif (close[i] < s3[i] and 
                  close[i] < ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below S3 (reversal) or 1d trend breaks
            if close[i] < s3[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above R3 (reversal) or 1d trend breaks
            if close[i] > r3[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals