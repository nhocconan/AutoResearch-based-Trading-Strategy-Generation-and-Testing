# 12h_DailyPivot_R2S2_1dATR_Trend_Volume
# Hypothesis: 12h strategy using daily pivot points (R2/S2) with 1d ATR trend filter and volume confirmation.
# In uptrend (ATR-based), buy breakouts above daily R2; in downtrend, sell breakdowns below daily S2.
# Uses 1d ATR to define trend strength (price > SMA + ATR = uptrend, price < SMA - ATR = downtrend).
# Daily R2/S2 provide stronger institutional support/resistance than R1/S1, reducing false breakouts.
# Volume confirms breakout strength. Target: 15-37 trades/year for 12h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for pivot points and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot points (using prior day's H/L/C)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R2 = Pivot + (High - Low)
    r2_1d = pivot_1d + (high_1d - low_1d)
    # S2 = Pivot - (High - Low)
    s2_1d = pivot_1d - (high_1d - low_1d)
    
    # Calculate 14-period ATR for trend filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate SMA20 for trend reference
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    
    # Define trend: Uptrend if close > SMA + ATR, Downtrend if close < SMA - ATR
    trend_up = close_1d > (sma_20 + atr_14)
    trend_down = close_1d < (sma_20 - atr_14)
    
    # Align daily indicators to 12h timeframe (wait for daily bar to close)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down.astype(float))
    
    # 12h volume confirmation (volume spike > 2.0x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        is_uptrend = trend_up_aligned[i] > 0.5
        is_downtrend = trend_down_aligned[i] > 0.5
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 2.0  # Volume spike filter for quality
        
        if position == 0:
            # Enter long: price breaks above daily R2 + uptrend + volume spike
            if (price_close > r2_aligned[i] and 
                is_uptrend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below daily S2 + downtrend + volume spike
            elif (price_close < s2_aligned[i] and 
                  is_downtrend and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal or opposite breakout
            if position == 1:
                if (not is_uptrend) or (price_close < s2_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (not is_downtrend) or (price_close > r2_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_DailyPivot_R2S2_1dATR_Trend_Volume"
timeframe = "12h"
leverage = 1.0