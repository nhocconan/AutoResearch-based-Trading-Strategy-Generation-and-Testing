#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation.
# Works in bull markets via breakouts, in bear via trend-filtered mean reversion at bands.
# Low frequency (~20-40 trades/year) to minimize fee drag. Uses discrete position sizing.
name = "4h_donchian20_12h_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate ATR(14) for volatility filter and stop sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period high/low)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 20-period average volume
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > vol_ma[i] if not np.isnan(vol_ma[i]) else True
        
        # Volatility filter: only trade when ATR is above its 50-period average (avoid low volatility chop)
        atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr[i] > atr_ma[i] if not np.isnan(atr_ma[i]) else True
        
        if position == 1:  # Long position
            # Exit: price touches opposite band OR trend changes OR volatility drops
            if close[i] <= lowest_low[i] or close[i] < ema_12h_aligned[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price touches opposite band OR trend changes OR volatility drops
            if close[i] >= highest_high[i] or close[i] > ema_12h_aligned[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above upper band + uptrend + volume confirmation + volatility filter
            if close[i] > highest_high[i] and close[i] > ema_12h_aligned[i] and vol_confirm and vol_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower band + downtrend + volume confirmation + volatility filter
            elif close[i] < lowest_low[i] and close[i] < ema_12h_aligned[i] and vol_confirm and vol_filter:
                position = -1
                signals[i] = -0.25
    
    return signals