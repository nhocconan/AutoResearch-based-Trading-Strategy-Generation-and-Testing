# 6h Ehlers Fisher Transform + 1d ATR Volatility Regime + Volume Spike
# Fisher Transform (period=8) identifies turning points in price
# Long when Fisher > trigger AND ATR(14) > ATR(50) (volatile regime) AND volume spike
# Short when Fisher < trigger AND ATR(14) > ATR(50) (volatile regime) AND volume spike
# Exit when Fisher crosses back through zero
# Works in both bull/bear by catching reversals in volatile markets
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "6h_FisherTransform_ATRVolRegime_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ehlers Fisher Transform on close prices (period=8)
    price = close
    # Normalize price over period
    period = 8
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    # Avoid division by zero
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    # Normalized price [-1, 1]
    normalized_price = 2 * ((price - lowest_low) / price_range) - 1
    # Clamp to avoid domain issues in log
    normalized_price = np.clip(normalized_price, -0.999, 0.999)
    
    # Fisher Transform formula
    fisher = 0.5 * np.log((1 + normalized_price) / (1 - normalized_price))
    # Smooth with 2-period EMA
    fisher_smoothed = pd.Series(fisher).ewm(span=2, adjust=False).mean().values
    
    # Trigger line (1-period delay of smoothed Fisher)
    trigger = np.roll(fisher_smoothed, 1)
    trigger[0] = 0
    
    # Calculate 1d ATR for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # True Range for daily data
    prev_close_1d = df_1d['close'].shift(1)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - prev_close_1d)
    tr3 = np.abs(df_1d['low'] - prev_close_1d)
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50) on daily
    atr14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr50_1d = pd.Series(tr_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volatility regime: short-term ATR > long-term ATR (expanding volatility)
    vol_regime = atr14_1d > atr50_1d
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Volume confirmation: current volume > 2.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for ATR calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(fisher_smoothed[i]) or np.isnan(trigger[i]) or 
            np.isnan(vol_regime_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Fisher crosses above trigger, volatile regime, volume spike
            if (fisher_smoothed[i] > trigger[i] and 
                fisher_smoothed[i-1] <= trigger[i-1] and  # Cross just happened
                vol_regime_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Fisher crosses below trigger, volatile regime, volume spike
            elif (fisher_smoothed[i] < trigger[i] and 
                  fisher_smoothed[i-1] >= trigger[i-1] and  # Cross just happened
                  vol_regime_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Fisher crosses below zero (mean reversion)
            if fisher_smoothed[i] < 0 and fisher_smoothed[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Fisher crosses above zero (mean reversion)
            if fisher_smoothed[i] > 0 and fisher_smoothed[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals