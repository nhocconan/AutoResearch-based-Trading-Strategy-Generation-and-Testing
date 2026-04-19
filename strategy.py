#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Donchian breakout with weekly volume confirmation and 1d ATR filter.
# Long when price breaks above 10-week Donchian high AND weekly volume > 1.8x 4-week average volume AND daily ATR(14) < daily ATR(50)
# Short when price breaks below 10-week Donchian low AND weekly volume > 1.8x 4-week average volume AND daily ATR(14) < daily ATR(50)
# Exit when price crosses back through the Donchian midpoint
# Uses weekly Donchian for trend structure, volume for confirmation, daily ATR regime filter to avoid chop.
# Target: 10-20 trades/year per symbol (30-80 total over 4 years).
name = "1d_WeeklyDonchian_Volume_ATRRegime"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 70:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian and volume
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate 10-week Donchian channels
    high_roll_weekly = pd.Series(df_weekly['high']).rolling(window=10, min_periods=10).max().values
    low_roll_weekly = pd.Series(df_weekly['low']).rolling(window=10, min_periods=10).min().values
    donchian_high = high_roll_weekly
    donchian_low = low_roll_weekly
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 4-week average volume for confirmation
    vol_ma_4w = pd.Series(df_weekly['volume']).rolling(window=4, min_periods=4).mean().values
    
    # Get daily ATR for regime filter (14 and 50 periods)
    df_1d = get_htf_data(prices, '1d')
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_weekly, donchian_mid)
    vol_ma_4w_aligned = align_htf_to_ltf(prices, df_weekly, vol_ma_4w)
    
    # Align daily ATR to daily timeframe (no change, but for consistency)
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(10, 4, 50)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma_4w_aligned[i]) or
            np.isnan(atr14_aligned[i]) or np.isnan(atr50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_vol = volume[i]  # Use current day's volume as proxy for weekly volume
        vol_ma = vol_ma_4w_aligned[i]
        atr14_val = atr14_aligned[i]
        atr50_val = atr50_aligned[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        mid = donchian_mid_aligned[i]
        
        # Regime filter: only trade in low volatility (ATR14 < ATR50)
        vol_regime = atr14_val < atr50_val
        
        if position == 0:
            # Long entry: break above upper band + volume spike + low vol regime
            if price > upper and weekly_vol > 1.8 * vol_ma and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower band + volume spike + low vol regime
            elif price < lower and weekly_vol > 1.8 * vol_ma and vol_regime:
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