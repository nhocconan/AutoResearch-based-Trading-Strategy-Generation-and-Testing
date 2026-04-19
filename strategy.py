#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R1/S1 breakout with volume confirmation and daily volatility filter.
# Long when price breaks above R1 (resistance 1) AND volume > 1.5x daily average volume AND daily ATR(14) < ATR(50) (low volatility regime)
# Short when price breaks below S1 (support 1) AND volume > 1.5x daily average volume AND daily ATR(14) < ATR(50)
# Exit when price crosses back through the daily pivot point (PP)
# Uses Camarilla pivots for precise levels, volume for confirmation, volatility filter to avoid choppy markets.
# Target: 20-30 trades/year per symbol.
name = "4h_Camarilla_Pivot_Volume_VolatilityFilter"
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
    
    # Get daily data for pivot points, ATR, and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    
    # Calculate daily average volume for confirmation
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate daily pivot points (Camarilla)
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pp = typical_price.values
    r1 = pp + (df_1d['high'] - df_1d['low']).values * 1.1 / 12
    s1 = pp - (df_1d['high'] - df_1d['low']).values * 1.1 / 12
    
    # Align pivot levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr14_aligned[i]) or np.isnan(atr50_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr14_val = atr14_aligned[i]
        atr50_val = atr50_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        # Volatility filter: only trade in low volatility (ATR14 < ATR50)
        vol_regime = atr14_val < atr50_val
        
        if position == 0:
            # Long entry: break above R1 + volume spike + low vol regime
            if price > r1_val and vol > 1.5 * vol_ma and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short entry: break below S1 + volume spike + low vol regime
            elif price < s1_val and vol > 1.5 * vol_ma and vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below pivot point
            if price < pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot point
            if price > pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals