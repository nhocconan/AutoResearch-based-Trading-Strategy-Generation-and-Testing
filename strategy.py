# 1h_Confluence_RSI_Volume_Trend_Filter_v1
# Hypothesis: Combine RSI mean reversion with volume confirmation and 4h trend filter on 1h timeframe.
# Uses RSI(14) < 30 for long, > 70 for short with volume spike (1.5x 20-period avg) and 4h EMA50 trend alignment.
# Targets 15-30 trades/year by requiring confluence of oversold/overbought, volume, and trend.
# Works in bull markets via pullbacks to rising EMA and in bear markets via bounces from falling EMA.
# Exit on RSI crossing 50 (mean reversion completion) or opposite signal.

#!/usr/bin/env python3
name = "1h_Confluence_RSI_Volume_Trend_Filter_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08:00 - 20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure RSI and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold), above 4h EMA50 (uptrend), with volume spike
            if (rsi[i] < 30 and 
                close[i] > ema_50_4h_aligned[i] and   # 4h uptrend
                volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought), below 4h EMA50 (downtrend), with volume spike
            elif (rsi[i] > 70 and 
                  close[i] < ema_50_4h_aligned[i] and   # 4h downtrend
                  volume_spike):
                signals[i] = -0.20
                position = -1
        elif position != 0:
            # Exit: RSI crosses 50 (mean reversion completion) or opposite signal
            if position == 1:
                # Exit long when RSI >= 50 or bearish setup
                if rsi[i] >= 50 or (rsi[i] > 70 and close[i] < ema_50_4h_aligned[i] and volume_spike):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit short when RSI <= 50 or bullish setup
                if rsi[i] <= 50 or (rsi[i] < 30 and close[i] > ema_50_4h_aligned[i] and volume_spike):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals