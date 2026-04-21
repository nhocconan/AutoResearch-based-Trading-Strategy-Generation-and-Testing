# 4h_Camarilla_Pivot_R1_S1_Breakout_12hEMA50_Trend_VolumeS
# Hypothesis: Breakout above daily R1 or below S1 with 12h EMA50 trend filter and volume confirmation.
# Works in bull (R1 breakout) and bear (S1 breakdown) regimes.
# Uses 12h EMA50 for trend direction and 4h volume spike for confirmation.
# Target: 20-35 trades/year by requiring confluence of pivot breakout, trend, and volume.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla pivot points
    df_1d = get_htf_data(prices, '1d')
    # Load 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels (R1, S1) from daily OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla R1 and S1
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 4h volume moving average (20-period)
    vol_ma_4h = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema50_val = ema50_12h_aligned[i]
        vol_ma = vol_ma_4h[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma
        
        if position == 0:
            # Long: Close breaks above R1, price > EMA50 (uptrend), volume confirmation
            if close[i] > r1_val and close[i] > ema50_val and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1, price < EMA50 (downtrend), volume confirmation
            elif close[i] < s1_val and close[i] < ema50_val and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below S1 (reversal signal)
                if close[i] < s1_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above R1 (reversal signal)
                if close[i] > r1_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0