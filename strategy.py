# 4h_1d_Camarilla_R1S1_Breakout_Volume_TrendFilter
# Hypothesis: Camarilla pivot levels from daily timeframe provide strong support/resistance. 
# Breakouts above R1 or below S1 with volume confirmation and trend filter (price > 50 EMA on 4h) capture momentum.
# Works in bull/bear: Only take longs when price above 50 EMA and breaking R1, shorts when below 50 EMA and breaking S1.
# Uses 4h timeframe for execution, targeting 20-50 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_R1S1_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Camarilla levels from previous day (to avoid look-ahead) ===
    # Use previous day's OHLC to calculate today's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using previous day's data
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    # Where C, H, L are from previous day
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Set first day's values to NaN (no previous day)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # === 4h: 50-period EMA for trend filter ===
    close = prices['close'].values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all 1d Camarilla data to 4h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        # Get values
        close_val = close[i]
        ema_val = ema_50[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(r1_val) or np.isnan(s1_val) or
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above EMA (uptrend), breaks above R1, volume confirmation
            if (close_val > ema_val and  # Uptrend filter
                close_val > r1_val and   # Break above R1
                vol_ratio_val > 1.5):    # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price below EMA (downtrend), breaks below S1, volume confirmation
            elif (close_val < ema_val and  # Downtrend filter
                  close_val < s1_val and   # Break below S1
                  vol_ratio_val > 1.5):    # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops below EMA or breaks below S1 (reversal)
            if close_val < ema_val or close_val < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above EMA or breaks above R1 (reversal)
            if close_val > ema_val or close_val > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals