# The strategy implements a volatility breakout system using the Average True Range (ATR) as a dynamic threshold for identifying significant price movements on the 12h timeframe. It combines volatility breakouts with trend confirmation using a long-term Exponential Moving Average (EMA) and volume confirmation to filter false signals. The strategy is designed to capture strong directional moves that occur after periods of consolidation, which are common in both bull and bear markets as they often precede major trend continuations or reversals. The use of ATR ensures the breakout sensitivity adapts to changing market volatility, while the EMA filter ensures trades are taken in the direction of the higher timeframe trend. Volume confirmation adds validity to the breakout by ensuring participation. The 12h timeframe reduces noise and transaction costs, aligning with the goal of minimizing trade frequency to avoid fee drag.

#!/usr/bin/env python3
name = "12h_Volatility_Breakout_Trend_Filter"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate True Range and ATR(14) on 12h data
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for EMA trend filter (more stable trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA50 on daily close
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma, out=np.ones_like(volume), where=vol_ma!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period for ATR and EMA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge filter
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long condition: bullish breakout above ATR-based resistance with volume and trend alignment
            atr_resistance = close[i-1] + 0.5 * atr[i]  # Break above prior close + 0.5*ATR
            if (close[i] > atr_resistance and 
                volume_surge and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short condition: bearish breakdown below ATR-based support with volume and trend alignment
            elif (close[i] < close[i-1] - 0.5 * atr[i] and 
                  volume_surge and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price reverts to mean or trend fails
            if position == 1:
                # Exit long: price returns to prior close or breaks below EMA
                if (close[i] < close[i-1]) or (close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to prior close or breaks above EMA
                if (close[i] > close[i-1]) or (close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals