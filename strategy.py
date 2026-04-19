#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w trend filter and volume confirmation
# - 1w EMA(34) defines trend direction (long when close > EMA34, short when close < EMA34)
# - 1d volume > 1.8x 20-period average for conviction (avoid low-volume false breakouts)
# - Entry on pullback to EMA34 in trending direction with volume confirmation
# - Exit on opposite EMA34 touch or trend reversal
# - Position size: 0.25 (25%) to balance return and drawdown
# - Designed for low frequency (target: 10-25 trades/year) to minimize fee drag
# - Works in both bull and bear by following higher timeframe trend

name = "1d_EMA34_Trend_VolumePullback_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(34) for trend direction
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Pre-compute for efficiency
    ema_34_1d = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_34_1d[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.8x average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.8 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Look for long entry: uptrend (close > 1w EMA34) + pullback to EMA34 + volume
            if (close[i] > ema_34_1w_aligned[i] and 
                low[i] <= ema_34_1d[i] <= high[i] and  # touched EMA34 today
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (close < 1w EMA34) + pullback to EMA34 + volume
            elif (close[i] < ema_34_1w_aligned[i] and 
                  low[i] <= ema_34_1d[i] <= high[i] and  # touched EMA34 today
                  volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on touch of EMA34 from below or trend reversal
            if (high[i] >= ema_34_1d[i] or close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on touch of EMA34 from above or trend reversal
            if (low[i] <= ema_34_1d[i] or close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#%%