#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with daily volume confirmation and 1d ATR regime filter.
# Long when Williams %R < -80 (oversold) AND volume > 1.5x daily average volume AND ATR(14) < ATR(50) (low volatility regime)
# Short when Williams %R > -20 (overbought) AND volume > 1.5x daily average volume AND ATR(14) < ATR(50)
# Exit when Williams %R crosses back above -50 for longs or below -50 for shorts
# Williams %R identifies mean-reversion extremes, volume confirms institutional interest, ATR filter avoids choppy markets.
# Target: 12-30 trades/year per symbol on 12h timeframe.
name = "12h_WilliamsR_Volume_ATRRegime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - df_1d['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(50).values  # Handle division by zero
    
    # Get 1d ATR for regime filter (14 and 50 periods)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Get 1d average volume for confirmation
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 50, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(atr14_aligned[i]) or 
            np.isnan(atr50_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        williams_r_val = williams_r_aligned[i]
        atr14_val = atr14_aligned[i]
        atr50_val = atr50_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        
        # Regime filter: only trade in low volatility (ATR14 < ATR50)
        vol_regime = atr14_val < atr50_val
        
        if position == 0:
            # Long entry: oversold + volume spike + low vol regime
            if williams_r_val < -80 and vol > 1.5 * vol_ma and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short entry: overbought + volume spike + low vol regime
            elif williams_r_val > -20 and vol > 1.5 * vol_ma and vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses above -50
            if williams_r_val > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses below -50
            if williams_r_val < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals