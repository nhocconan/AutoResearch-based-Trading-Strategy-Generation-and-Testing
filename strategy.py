#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d volume confirmation and 1d ATR regime filter.
# Long when Jaw < Teeth < Lips (bullish alignment) AND volume > 1.3x daily average AND ATR(14) < ATR(50) (low volatility)
# Short when Jaw > Teeth > Lips (bearish alignment) AND volume > 1.3x daily average AND ATR(14) < ATR(50)
# Exit when Alligator lines cross (Jaw crosses Teeth) or volatility regime shifts.
# Uses Alligator for trend alignment, volume for confirmation, ATR regime to avoid choppy markets.
# Target: 20-30 trades/year per symbol.
name = "4h_Alligator_Volume_ATRRegime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and volume/ATR filters
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator: SMAs with specific offsets (13,8,5 smoothed by 8,5,3)
    jaw = pd.Series(df_1d['close']).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(df_1d['close']).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1d ATR for regime filter (14 and 50 periods)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    
    # Get 1d average volume for confirmation (20-period MA)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13+8, 8+5, 5+3, 14, 50, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(atr14_aligned[i]) or np.isnan(atr50_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        atr14_val = atr14_aligned[i]
        atr50_val = atr50_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        
        # Bullish alignment: Jaw < Teeth < Lips
        bullish_align = jaw_val < teeth_val < lips_val
        # Bearish alignment: Jaw > Teeth > Lips
        bearish_align = jaw_val > teeth_val > lips_val
        
        # Regime filter: only trade in low volatility (ATR14 < ATR50)
        vol_regime = atr14_val < atr50_val
        
        if position == 0:
            # Long entry: bullish alignment + volume spike + low vol regime
            if bullish_align and vol > 1.3 * vol_ma and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment + volume spike + low vol regime
            elif bearish_align and vol > 1.3 * vol_ma and vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator lines cross (Jaw crosses above Teeth) OR volatility regime shifts
            if jaw_val > teeth_val or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator lines cross (Jaw crosses below Teeth) OR volatility regime shifts
            if jaw_val < teeth_val or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals