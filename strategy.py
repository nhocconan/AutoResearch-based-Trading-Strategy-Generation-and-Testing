# #!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses Camarilla pivot levels (R3/S3) from daily timeframe for breakout entries,
filtered by 1-day trend (EMA34) and volume spike (2x average volume). Exits on opposite Camarilla level (S3/R3) touch.
Designed for low trade frequency (20-50/year) with strong edge in both bull and bear markets by
trading intraday breakouts aligned with daily trend. Camarilla levels provide precise support/resistance
based on prior day's range, effective in ranging and trending markets.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla calculation and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla formula: 
    # H = (High - Low) * 1.1 / 2 + Close
    # L = Close - (High - Low) * 1.1 / 2
    # S3 = L - (H - L) * 0.5
    # R3 = H + (H - L) * 0.5
    # S4 = L - (H - L) * 1.0
    # R4 = H + (H - L) * 1.0
    
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate daily range
    daily_range = prev_high - prev_low
    
    # Camarilla levels
    H = prev_close + daily_range * 1.1 / 2
    L = prev_close - daily_range * 1.1 / 2
    
    S3 = L - (H - L) * 0.5
    R3 = H + (H - L) * 0.5
    
    # Align Camarilla levels to 4h timeframe (available after daily candle close)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    
    # Daily trend filter (EMA34)
    ema_34_1d = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (2x average volume)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(S3_4h[i]) or np.isnan(R3_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: price breaks above R3 + above daily EMA34 + volume spike
            if (close[i] > R3_4h[i] and 
                close[i] > ema_34_4h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + below daily EMA34 + volume spike
            elif (close[i] < S3_4h[i] and 
                  close[i] < ema_34_4h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to S3 level
                if close[i] <= S3_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to R3 level
                if close[i] >= R3_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals