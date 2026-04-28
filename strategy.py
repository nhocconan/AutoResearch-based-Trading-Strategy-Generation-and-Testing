# 4h_Stochastic_Momentum_Index_Signal_With_Volume_Confirmation
# Hypothesis: SMI captures overbought/oversold momentum with less whipsaw than RSI. 
# Uses SMI crosses at extreme levels (< -40 for long, > 40 for short) with volume confirmation and 
# 1-day trend filter to align with higher timeframe momentum. Designed for 4h timeframe to 
# balance trade frequency (~25-40 trades/year) and signal quality. Works in bull/bear markets 
# by taking both long and short signals based on momentum extremes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Stochastic Momentum Index (SMI) with length=10, smooth_k=3, smooth_d=3
    # SMI = (Close - Median of HL range) / (0.5 * Range of HL) * 100, then smoothed
    hl_range = high - low
    hh = pd.Series(high).rolling(window=10, min_periods=10).max().values
    ll = pd.Series(low).rolling(window=10, min_periods=10).min().values
    diff = close - (hh + ll) / 2.0  # Distance from midpoint of HL range
    abs_diff = np.abs(diff)
    hl_range_sum = pd.Series(hh - ll).rolling(window=10, min_periods=10).sum().values
    
    # Avoid division by zero
    smi_raw = np.where(hl_range_sum != 0, (diff * 2) / hl_range_sum * 100, 0)
    
    # Smooth with double smoothing (3-period SMA twice)
    smi_k = pd.Series(smi_raw).rolling(window=3, min_periods=3).mean().values
    smi_d = pd.Series(smi_k).rolling(window=3, min_periods=3).mean().values
    
    # Calculate volume spike (>1.5x 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(smi_d[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 1-day EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # SMI signals at extreme levels
        smi_long_signal = smi_d[i] < -40 and smi_d[i-1] >= -40  # Cross above -40
        smi_short_signal = smi_d[i] > 40 and smi_d[i-1] <= 40   # Cross below 40
        
        # Entry logic:
        # Long: SMI crosses above -40 (oversold recovery) with volume and uptrend
        long_entry = vol_confirm and trend_up and smi_long_signal
        # Short: SMI crosses below 40 (overbought rejection) with volume and downtrend
        short_entry = vol_confirm and trend_down and smi_short_signal
        
        # Exit logic: Opposite SMI extreme or trend reversal
        long_exit = smi_d[i] > 20 or not trend_up  # Exit when SMI rises above 20 or trend down
        short_exit = smi_d[i] < -20 or not trend_down  # Exit when SMI falls below -20 or trend up
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Stochastic_Momentum_Index_Signal_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0