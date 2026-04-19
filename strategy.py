#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_ChaikinMoneyFlow_Breakout_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Chaikin Money Flow (CMF) - 20 period
    # CMF = sum of MFV over period / sum of volume over period
    # MFV = Volume * ((Close - Low) - (High - Close)) / (High - Low)
    # When High == Low, MFV = 0 to avoid division by zero
    
    high_low = high_1d - low_1d
    # Avoid division by zero
    high_low_safe = np.where(high_low == 0, 1, high_low)
    money_flow_multiplier = ((close_1d - low_1d) - (high_1d - close_1d)) / high_low_safe
    money_flow_volume = money_flow_multiplier * volume_1d
    
    # Calculate CMF(20)
    mfv_sum = pd.Series(money_flow_volume).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume_1d).rolling(window=20, min_periods=20).sum().values
    cmf = np.divide(mfv_sum, vol_sum, out=np.zeros_like(mfv_sum), where=vol_sum!=0)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align CMF and EMA to 4h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_1d, cmf)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 4h Donchian Channel (20-period)
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(cmf_aligned[i]) or np.isnan(ema_50_aligned[i]) or \
           np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        cmf_val = cmf_aligned[i]
        ema_50_val = ema_50_aligned[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian upper + CMF positive (>0.1) + price above EMA50
            if price > donchian_upper[i] and cmf_val > 0.1 and price > ema_50_val and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + CMF negative (<-0.1) + price below EMA50
            elif price < donchian_lower[i] and cmf_val < -0.1 and price < ema_50_val and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below Donchian upper OR CMF turns negative
            if price < donchian_upper[i] or cmf_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above Donchian lower OR CMF turns positive
            if price > donchian_lower[i] or cmf_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals