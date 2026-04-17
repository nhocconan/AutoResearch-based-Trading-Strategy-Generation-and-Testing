#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using weekly Keltner Channel breakout with volume confirmation and trend filter.
- Enter long when price breaks above upper KC(20,2) + volume > 1.5x 20-period volume MA + price above 20 EMA
- Enter short when price breaks below lower KC(20,2) + volume > 1.5x 20-period volume MA + price below 20 EMA
- Exit when price crosses back inside Keltner Channels
- Fixed position size 0.25 to manage drawdown
- Uses volatility-based channel breakouts, effective in trending markets
- Weekly timeframe provides stable trend filter for daily entries
"""

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
    
    # Keltner Channels (20, 2)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean()
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean()
    upper_kc = ema_20 + 2 * atr
    lower_kc = ema_20 - 2 * atr
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Trend filter: 20 EMA
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for 20 EMA and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20.iloc[i]) or np.isnan(atr.iloc[i]) or 
            np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        upper = upper_kc.iloc[i]
        lower = lower_kc.iloc[i]
        ema_val = ema_20.iloc[i]
        
        if position == 0:
            # Look for Keltner Channel breakouts with volume confirmation and trend filter
            # Long: price breaks above upper KC + volume spike + price above 20 EMA
            if price > upper and vol > 1.5 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower KC + volume spike + price below 20 EMA
            elif price < lower and vol > 1.5 * vol_ma and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses back inside Keltner Channels (mean reversion)
            if price < upper and price > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses back inside Keltner Channels (mean reversion)
            if price < upper and price > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KeltnerChannelBreakout_VolumeTrend"
timeframe = "1d"
leverage = 1.0