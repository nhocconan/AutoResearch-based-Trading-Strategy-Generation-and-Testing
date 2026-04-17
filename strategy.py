# 1h Multi-Timeframe Strategy: 4h Trend + 1d Momentum + Volume
# Uses 4h EMA50 for trend direction, 1d RSI for momentum, volume filter for confirmation
# Entry only during 08-20 UTC session to reduce noise
# Target: 20-35 trades/year per symbol (80-140 total over 4 years)
# Signal size: 0.20 (20% of capital)
# Exit on opposite signal or stop via signal=0 when momentum reverses

#!/usr/bin/env python3
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
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # === 4h EMA50 for trend direction ===
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d RSI(14) for momentum ===
    df_1d = get_htf_data(prices, '1d')
    rsi_period = 14
    delta = pd.Series(df_1d['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period, min_periods=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = gain / loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # === Volume filter: 1h volume > 1.5x 20-period MA ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC only
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
            
        # Skip if any data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: uptrend (price > 4h EMA50) + bullish momentum (RSI > 50) + volume
            if (price > ema_50_4h_aligned[i] and 
                rsi_1d_aligned[i] > 50 and 
                vol > 1.5 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
                
            # Short: downtrend (price < 4h EMA50) + bearish momentum (RSI < 50) + volume
            elif (price < ema_50_4h_aligned[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  vol > 1.5 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long exit: trend break or momentum fade
            if (price < ema_50_4h_aligned[i] or 
                rsi_1d_aligned[i] < 40):  # RSI drop signals weakening momentum
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short exit: trend break or momentum fade
            if (price > ema_50_4h_aligned[i] or 
                rsi_1d_aligned[i] > 60):  # RSI rise signals weakening short momentum
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hEMA50_1dRSI_Volume_Session"
timeframe = "1h"
leverage = 1.0