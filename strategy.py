#!/usr/bin/env python3
"""
12h_RSI_Overbought_Oversold_MeanReversion_v1
Hypothesis: On 12h timeframe, use RSI(14) for mean reversion entries, filtered by daily EMA trend and volume confirmation. 
Long when RSI < 30 (oversold) with daily uptrend and volume spike. Short when RSI > 70 (overbought) with daily downtrend and volume spike.
Exits when RSI returns to 50 (neutral). This strategy targets reversals in both bull and bear markets, 
with fewer trades due to strict RSI thresholds and volume confirmation, minimizing fee drag on 12h timeframe.
"""
name = "12h_RSI_Overbought_Oversold_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    close_1d = pd.Series(df_1d['close'])
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align daily RSI to 12h timeframe (wait for daily close)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Daily EMA34 for trend filter
    ema_34 = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 2.0 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 5 bars between trades (60 hours on 12h TF) to reduce frequency
            if bars_since_exit < 5:
                continue
                
            # Long: RSI < 30 (oversold) + daily uptrend + volume filter
            if (rsi_aligned[i] < 30 and 
                close[i] > ema_34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: RSI > 70 (overbought) + daily downtrend + volume filter
            elif (rsi_aligned[i] > 70 and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: RSI returns to 50 (neutral)
            if not np.isnan(rsi_aligned[i]):
                if position == 1 and rsi_aligned[i] >= 50:
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                elif position == -1 and rsi_aligned[i] <= 50:
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    # Hold position
                    signals[i] = 0.25 if position == 1 else -0.25
            else:
                # Hold if RSI not ready
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals