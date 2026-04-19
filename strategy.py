#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Daily trend following with weekly EMA34 trend filter and volume confirmation.
# Long when: Price > Weekly EMA34 AND volume > 1.5x 20-period average
# Short when: Price < Weekly EMA34 AND volume > 1.5x 20-period average
# Exit when: Price crosses back below/above Weekly EMA34
# Weekly EMA34 filters direction, volume confirms strength, daily price action triggers entries.
# Works in trending markets by capturing sustained moves. Target: 10-20 trades/year per symbol.
name = "1d_EMA34_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w EMA34 ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema34_1w_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Price > Weekly EMA34 + volume spike
            if price > ema34 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: Price < Weekly EMA34 + volume spike
            elif price < ema34 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below Weekly EMA34
            if price < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above Weekly EMA34
            if price > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals