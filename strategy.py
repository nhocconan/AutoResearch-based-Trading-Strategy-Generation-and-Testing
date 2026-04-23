#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike.
Long when price breaks above R3 AND close > 1w EMA34 AND volume > 2.0x average.
Short when price breaks below S3 AND close < 1w EMA34 AND volume > 2.0x average.
Exit when price returns to Camarilla Pivot (PP) level or volume drops below average.
Camarilla levels provide precise intraday support/resistance, 1w EMA34 filters for higher timeframe trend,
volume spike confirms conviction. Designed for 1d timeframe targeting 30-100 total trades over 4 years
with low frequency to minimize fee drag. Works in both bull and bear markets by only taking trades
aligned with 1w trend and using mean-reversion exit at pivot.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for EMA34 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w data
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # We need previous day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First value will be NaN due to roll, that's fine
    
    pp = (prev_high + prev_low + prev_close) / 3.0
    r3 = prev_close + (prev_high - prev_low) * 1.1 / 2.0
    s3 = prev_close - (prev_high - prev_low) * 1.1 / 2.0
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(pp[i]) or np.isnan(r3[i]) or 
            np.isnan(s3[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1w_aligned[i]
        pp_val = pp[i]
        r3_val = r3[i]
        s3_val = s3[i]
        price = close[i]
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 AND close > 1w EMA34 AND volume spike
            if (price > r3_val and close[i] > ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND close < 1w EMA34 AND volume spike
            elif (price < s3_val and close[i] < ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to pivot level OR volume drops below average
                if (price <= pp_val or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to pivot level OR volume drops below average
                if (price >= pp_val or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Camarilla_R3S3_Breakout_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0