# Hypothesis: 1h Bollinger Band squeeze breakout with 4h trend filter and volume confirmation.
# Bollinger Band squeeze (low volatility) precedes explosive moves. Breakout from squeeze
# captures momentum. 4h EMA(50) provides trend filter to avoid counter-trend trades.
# Volume > 2x average confirms institutional participation. Works in bull/bear as 4h EMA adapts.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).
# Timeframe: 1h, HTF: 4h

#!/usr/bin/env python3
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
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA(50) for trend filter
    ema_len = 50
    if len(df_4h) < ema_len:
        return np.zeros(n)
    
    ema_4h = pd.Series(df_4h['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Bollinger Bands (20, 2) on 1h
    bb_len = 20
    bb_std = 2
    if len(close) < bb_len:
        return np.zeros(n)
    
    ma = pd.Series(close).rolling(window=bb_len, min_periods=bb_len).mean().values
    std = pd.Series(close).rolling(window=bb_len, min_periods=bb_len).std().values
    upper = ma + bb_std * std
    lower = ma - bb_std * std
    
    # Bollinger Band width (normalized by MA) for squeeze detection
    bb_width = (upper - lower) / ma
    # Squeeze threshold: BB width below 20-period percentile (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma  # True when in low volatility state
    
    # Volume confirmation: 2x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(50, bb_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(upper[i]) or
            np.isnan(lower[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(squeeze[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 4h EMA50
        above_ema = close[i] > ema_4h_aligned[i]
        below_ema = close[i] < ema_4h_aligned[i]
        
        # Volume confirmation: current volume > 2x average
        volume_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Enter long: Bollinger breakout above upper band + above 4h EMA + volume + squeeze
            if (close[i] > upper[i] and 
                above_ema and 
                volume_confirmed and 
                squeeze[i]):
                position = 1
                signals[i] = position_size
            # Enter short: Bollinger breakdown below lower band + below 4h EMA + volume + squeeze
            elif (close[i] < lower[i] and 
                  below_ema and 
                  volume_confirmed and 
                  squeeze[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle band or breaks below lower band
            if close[i] < ma[i] or close[i] < lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle band or breaks above upper band
            if close[i] > ma[i] or close[i] > upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_Bollinger_Squeeze_Breakout_4hEMA50_Volume_v1"
timeframe = "1h"
leverage = 1.0