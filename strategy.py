#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d RSI for mean reversion and 4h volume spike for confirmation.
# RSI < 30 triggers long entries, RSI > 70 triggers short entries.
# Volume > 2.0x 20-period average confirms the reversal momentum.
# Only trades in direction of 4h EMA(50) trend filter to avoid counter-trend trades.
# Designed for mean reversion in ranging markets while avoiding strong trends.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA(50) for trend filter
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate RSI(14) on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: 2.0x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)  # Need EMA(50) and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_50[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Look for mean reversion entries
            # Long: RSI < 30 (oversold) AND price above EMA(50) (uptrend filter)
            if (rsi_aligned[i] < 30 and 
                close[i] > ema_50[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: RSI > 70 (overbought) AND price below EMA(50) (downtrend filter)
            elif (rsi_aligned[i] > 70 and 
                  close[i] < ema_50[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or price crosses below EMA(50)
            if (rsi_aligned[i] >= 50 or 
                close[i] < ema_50[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or price crosses above EMA(50)
            if (rsi_aligned[i] <= 50 or 
                close[i] > ema_50[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dRSI_VolumeSpike_EMAFilter_v1"
timeframe = "4h"
leverage = 1.0