#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Donchian breakout with weekly volume confirmation and weekly ATR filter.
# Long when price breaks above 10-week Donchian high AND weekly volume > 1.5x 20-week average volume AND ATR(4) < ATR(10) (low volatility regime)
# Short when price breaks below 10-week Donchian low AND weekly volume > 1.5x 20-week average volume AND ATR(4) < ATR(10)
# Exit when price crosses back through the Donchian midpoint
# Uses weekly Donchian for trend following structure, weekly volume for confirmation, weekly ATR regime filter to avoid chop.
# Target: 15-25 trades/year per symbol.
name = "1d_WeeklyDonchian_Volume_ATRRegime"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for indicators
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly ATR (4 and 10 periods) for regime filter
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr4 = pd.Series(tr).rolling(window=4, min_periods=4).mean().values
    atr10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr4_aligned = align_htf_to_ltf(prices, df_1w, atr4)
    atr10_aligned = align_htf_to_ltf(prices, df_1w, atr10)
    
    # Get weekly average volume for confirmation (20-period)
    vol_ma_1w = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Calculate weekly Donchian channels (10-period)
    high_roll = pd.Series(df_1w['high']).rolling(window=10, min_periods=10).max().values
    low_roll = pd.Series(df_1w['low']).rolling(window=10, min_periods=10).min().values
    donchian_mid = (high_roll + low_roll) / 2
    high_roll_aligned = align_htf_to_ltf(prices, df_1w, high_roll)
    low_roll_aligned = align_htf_to_ltf(prices, df_1w, low_roll)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr4_aligned[i]) or np.isnan(atr10_aligned[i]) or 
            np.isnan(vol_ma_1w_aligned[i]) or np.isnan(high_roll_aligned[i]) or 
            np.isnan(low_roll_aligned[i]) or np.isnan(donchian_mid_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr4_val = atr4_aligned[i]
        atr10_val = atr10_aligned[i]
        vol_ma = vol_ma_1w_aligned[i]
        vol = volume[i]
        upper = high_roll_aligned[i]
        lower = low_roll_aligned[i]
        mid = donchian_mid_aligned[i]
        
        # Regime filter: only trade in low volatility (ATR4 < ATR10)
        vol_regime = atr4_val < atr10_val
        
        if position == 0:
            # Long entry: break above upper band + volume spike + low vol regime
            if price > upper and vol > 1.5 * vol_ma and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower band + volume spike + low vol regime
            elif price < lower and vol > 1.5 * vol_ma and vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below midpoint
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above midpoint
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals