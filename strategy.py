# 12h_Camarilla_R1_S1_Breakout_With_Volume_and_Trend
# Hypothesis: Breakout above R1 or below S1 using daily pivot levels with volume confirmation and daily trend filter.
# Exit at opposite pivot level. Uses tighter volume filter (2.0) and longer lookback to reduce trades and improve quality.
# Target: 12-37 trades/year on 12h timeframe for BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Camarilla pivot levels from previous day ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price
    pp = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    r1 = pp + (range_hl * 1.1 / 12)
    s1 = pp - (range_hl * 1.1 / 12)
    
    # Align to 12h timeframe (previous day's levels available at open)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Daily EMA34 for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume confirmation (20-period average) - stricter threshold ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: price breaks above R1 + uptrend + strong volume
            if (price_close > r1_level and
                price_close > ema_trend and
                vol_ratio_val > 2.0):  # Stricter volume filter
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + downtrend + strong volume
            elif (price_close < s1_level and
                  price_close < ema_trend and
                  vol_ratio_val > 2.0):  # Stricter volume filter
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price reaches opposite pivot level
            if position == 1 and price_close < s1_level:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > r1_level:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_With_Volume_and_Trend"
timeframe = "12h"
leverage = 1.0