# 1h_4h_1d_Camarilla_Pivot_Trend_With_Volume
# Hypothesis: Uses 4h Camarilla pivot levels (S1, S2, R1, R2) for entry/exit with 1d trend filter and volume confirmation.
# Long when price breaks above R1 with volume and price above 1d EMA34.
# Short when price breaks below S1 with volume and price below 1d EMA34.
# Exit when price touches opposite S2/R2 level or loses volume confirmation.
# Designed for 1h timeframe with 4h directional bias and 1d trend filter to reduce whipsaw.
# Works in both bull and bear markets by following the intermediate-term trend and using volatility-based pivot levels.

name = "1h_4h_1d_Camarilla_Pivot_Trend_With_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 4h Camarilla Pivot Levels (using previous 4h bar) ---
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate pivot and levels from previous 4h bar
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    
    # Camarilla levels: S1 = close - (range * 1.1/6), R1 = close + (range * 1.1/6)
    s1_4h = close_4h - (range_4h * 1.1 / 6)
    r1_4h = close_4h + (range_4h * 1.1 / 6)
    s2_4h = close_4h - (range_4h * 1.1 / 4)
    r2_4h = close_4h + (range_4h * 1.1 / 4)
    
    # Align 4h levels to 1h (wait for 4h bar to close)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s2_4h_aligned = align_htf_to_ltf(prices, df_4h, s2_4h)
    r2_4h_aligned = align_htf_to_ltf(prices, df_4h, r2_4h)
    
    # --- 1d Trend Filter (EMA34 on close) ---
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume Spike Detection (24-period average on 1h) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or
            np.isnan(s2_4h_aligned[i]) or np.isnan(r2_4h_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above R1 with volume, above 1d EMA34
            if (close[i] > r1_4h_aligned[i] and 
                volume_spike and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with volume, below 1d EMA34
            elif (close[i] < s1_4h_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: opposite S2/R2 level or loss of volume/momentum
            if position == 1:
                # Exit long: price touches S2 or loses volume/upside momentum
                if (close[i] < s2_4h_aligned[i] or 
                    not volume_spike or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price touches R2 or loses volume/downside momentum
                if (close[i] > r2_4h_aligned[i] or 
                    not volume_spike or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals