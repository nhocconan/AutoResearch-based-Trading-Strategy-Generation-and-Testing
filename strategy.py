#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above R4 AND price > 1d EMA50 AND volume > 2.0x 24-bar average.
# Short when price breaks below S4 AND price < 1d EMA50 AND volume > 2.0x 24-bar average.
# Exit when price crosses the Camarilla pivot point (PP).
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
# R4/S4 levels provide stronger breakout signals than R3/S3, reducing false entries.
# Volume spike threshold increased to 2.0x for stricter confirmation.
# Works in bull/bear via 1d EMA50 trend filter and volume spike to avoid false breakouts.

name = "12h_Camarilla_R4S4_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    pp = (high_1d + low_1d + close_1d_vals) / 3.0
    r4 = pp + (high_1d - low_1d) * 1.1 / 2.0 * 2.0  # R4 = PP + 2*(H-L)*1.1/2
    s4 = pp - (high_1d - low_1d) * 1.1 / 2.0 * 2.0  # S4 = PP - 2*(H-L)*1.1/2
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 2.0x 24-period average (stricter for fewer trades)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above R4, uptrend (price > 1d EMA50), volume confirmation
            if (curr_high > r4_aligned[i] and 
                curr_close > ema_50_1d_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below S4, downtrend (price < 1d EMA50), volume confirmation
            elif (curr_low < s4_aligned[i] and 
                  curr_close < ema_50_1d_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below pivot point (PP)
            if curr_close < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above pivot point (PP)
            if curr_close > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals