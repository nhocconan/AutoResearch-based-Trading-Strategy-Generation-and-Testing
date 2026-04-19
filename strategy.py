#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d trend filter (EMA34) + volume confirmation.
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0, Bear Power < 0 (bullish divergence), price > 1d EMA34, volume > 1.5x daily average.
# Short when Bull Power < 0, Bear Power > 0 (bearish divergence), price < 1d EMA34, volume > 1.5x daily average.
# Exit when Elder Ray signals reverse or volume drops below average.
# Uses Elder Ray for momentum, 1d EMA for trend filter, volume for confirmation.
# Target: 12-37 trades/year per symbol.
name = "6h_ElderRay_EMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 1d average volume for confirmation
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 6-day EMA13 for Elder Ray (using 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 13)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema34_1d_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        bp = bull_power[i]
        br = bear_power[i]
        
        # Volume confirmation
        vol_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: bullish Elder Ray (BP>0, BR<0), price above 1d EMA34, volume spike
            if bp > 0 and br < 0 and price > ema34 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish Elder Ray (BP<0, BR>0), price below 1d EMA34, volume spike
            elif bp < 0 and br > 0 and price < ema34 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Elder Ray turns bearish or volume drops
            if bp <= 0 or br >= 0 or not vol_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Elder Ray turns bullish or volume drops
            if bp >= 0 or br <= 0 or not vol_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals