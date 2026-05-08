#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_EquiVol_Donchian_Breakout_Strategy"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for volatility calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily volatility: 10-day ATR normalized by price
    atr_10d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 10:
            atr_10d[i] = np.mean(atr_10d[:i+1]) if i > 0 else tr
        else:
            atr_10d[i] = (atr_10d[i-1] * 9 + tr) / 10
    
    # Normalized volatility (ATR/price)
    vol_norm = atr_10d / close_1d
    vol_norm = np.nan_to_num(vol_norm, nan=0.0)
    
    # Equal volatility weight: inverse of normalized vol
    eq_vol_weight = 1.0 / (vol_norm + 1e-8)
    # Cap extreme weights
    eq_vol_weight = np.clip(eq_vol_weight, 0.5, 3.0)
    
    # Align daily volatility weight to 4h timeframe
    eq_vol_weight_aligned = align_htf_to_ltf(prices, df_1d, eq_vol_weight)
    
    # 4h Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # 4h EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(eq_vol_weight_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(ema50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_weight = eq_vol_weight_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND above EMA50
            long_cond = (close[i] > donch_high[i] and 
                        close[i] > ema50[i])
            
            # Short: price breaks below Donchian low AND below EMA50
            short_cond = (close[i] < donch_low[i] and 
                         close[i] < ema50[i])
            
            if long_cond:
                # Scale position by volatility weight (0.25 base)
                pos_size = 0.25 * vol_weight
                pos_size = min(pos_size, 0.35)  # Cap at 35%
                signals[i] = pos_size
                position = 1
            elif short_cond:
                # Scale position by volatility weight (0.25 base)
                pos_size = 0.25 * vol_weight
                pos_size = min(pos_size, 0.35)  # Cap at 35%
                signals[i] = -pos_size
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian low OR below EMA50
            if close[i] < donch_low[i] or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 * vol_weight
                signals[i] = min(signals[i], 0.35)
        elif position == -1:
            # Short exit: price closes above Donchian high OR above EMA50
            if close[i] > donch_high[i] or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25 * vol_weight
                signals[i] = max(signals[i], -0.35)
    
    return signals
#%%