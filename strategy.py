#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) extreme reversion with 4h EMA50 trend filter and volume spike confirmation (>2.0x 20-period average)
# RSI extremes (<30 for long, >70 for short) indicate exhaustion; 4h EMA50 ensures alignment with higher timeframe trend
# Volume spike confirms institutional participation; discrete sizing (0.20) minimizes fee churn
# Works in both bull/bear markets: mean reversion at extremes + trend filter avoids counter-trend trades in strong moves
# Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe

name = "1h_RSI_Extreme_Reversion_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # RSI(14) on 1h timeframe
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 20-period average volume for confirmation (on 1h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20)  # 4h EMA50, RSI, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_4h = ema_50_4h_aligned[i]
        curr_rsi = rsi_values[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (50) OR volume spike ends
            if curr_rsi >= 50 or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (50) OR volume spike ends
            if curr_rsi <= 50 or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: RSI < 30 (oversold) + above 4h EMA50 + volume confirmation
            if (curr_rsi < 30 and 
                curr_close > curr_ema_4h and 
                vol_confirm):
                signals[i] = 0.20
                position = 1
            # Short entry: RSI > 70 (overbought) + below 4h EMA50 + volume confirmation
            elif (curr_rsi > 70 and 
                  curr_close < curr_ema_4h and 
                  vol_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals