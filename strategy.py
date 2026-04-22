#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d EMA34 trend filter and volume spike
# Uses Donchian channel (20-period high/low) for breakout entries
# Long when price breaks above upper band with 1d uptrend and volume spike
# Short when price breaks below lower band with 1d downtrend and volume spike
# Designed for 12h timeframe to target 12-37 trades/year per symbol.
# Works in bull/bear via 1d trend filter and volatility-based volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for higher timeframe trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channel (20-period) on 12h data
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(n):
        if i >= lookback - 1:
            upper[i] = np.max(high[i-lookback+1:i+1])
            lower[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume spike filter (20-period on 12h data)
    vol_ma20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 20 - 1:
            vol_ma20[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer, higher quality trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper band + 1d uptrend + volume spike
            if (close[i] > upper[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band + 1d downtrend + volume spike
            elif (close[i] < lower[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to Donchian midline or trend reversal
            midline = (upper[i] + lower[i]) / 2.0
            
            if position == 1:
                # Exit on price below midline or trend reversal
                if (close[i] < midline or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on price above midline or trend reversal
                if (close[i] > midline or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0