#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ATR trend filter
# - Primary: 4h price breaks above/below 20-period Donchian channel for trend capture
# - HTF: 1d volume > 1.3x 20-period MA for confirmation (avoids low-volume breakouts)
# - Regime filter: 4h ATR(14) > 1.5x ATR(50) to ensure trending market (avoids chop)
# - Long: Close > Upper Donchian(20) + volume confirmation + ATR trending regime
# - Short: Close < Lower Donchian(20) + volume confirmation + ATR trending regime
# - Exit: Close crosses back inside Donchian channel OR ATR regime shifts to ranging
# - Position sizing: 0.25 (discrete level, balances return/drawdown, reduces fee churn)
# - Works in bull/bear: Donchian adapts to volatility, volume filters false breakouts, ATR regime avoids chop
# - Target: 80-150 total trades over 4 years (20-38/year) for 4h timeframe

name = "4h_1d_donchian_volume_atr_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:  # Need enough data for calculations
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # Pre-compute 1d data
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Donchian channel (20-period)
    upper_donchian = np.full(len(close_4h), np.nan)
    lower_donchian = np.full(len(close_4h), np.nan)
    
    for i in range(19, len(close_4h)):
        if not (np.isnan(high_4h[i-19:i+1]).any() or np.isnan(low_4h[i-19:i+1]).any()):
            upper_donchian[i] = np.max(high_4h[i-19:i+1])
            lower_donchian[i] = np.min(low_4h[i-19:i+1])
    
    # Calculate 4h ATR(14) and ATR(50) for regime filter
    atr_14 = np.full(len(close_4h), np.nan)
    atr_50 = np.full(len(close_4h), np.nan)
    tr = np.full(len(close_4h), np.nan)
    
    # True Range
    for i in range(1, len(close_4h)):
        if not (np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(close_4h[i-1])):
            tr[i] = max(
                high_4h[i] - low_4h[i],
                abs(high_4h[i] - close_4h[i-1]),
                abs(low_4h[i] - close_4h[i-1])
            )
    
    # ATR(14) using Wilder's smoothing
    for i in range(14, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            if i == 14:
                atr_14[i] = np.mean(tr[1:15])  # First ATR is simple average
            else:
                atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # ATR(50) using Wilder's smoothing
    for i in range(50, len(tr)):
        if not np.isnan(tr[i-49:i+1]).any():
            if i == 50:
                atr_50[i] = np.mean(tr[1:51])  # First ATR is simple average
            else:
                atr_50[i] = (atr_50[i-1] * 49 + tr[i]) / 50
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        if not np.isnan(volume_1d[i-19:i+1]).any():
            volume_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align all HTF/LTF indicators to 4h timeframe
    upper_donchian_aligned = align_htf_to_ltf(prices, prices, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, prices, lower_donchian)
    atr_14_aligned = align_htf_to_ltf(prices, prices, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, prices, atr_50)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period for ATR(50)
        # Skip if any required data is invalid
        if (np.isnan(upper_donchian_aligned[i]) or np.isnan(lower_donchian_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.3 * volume_ma_20_1d_aligned[i]
        
        # ATR regime filter: ATR(14) > 1.5x ATR(50) indicates trending market
        atr_trending = atr_14_aligned[i] > 1.5 * atr_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Close > Upper Donchian + volume confirmation + ATR trending
            if close_4h[i] > upper_donchian_aligned[i] and volume_confirm and atr_trending:
                position = 1
                signals[i] = 0.25
            # Short entry: Close < Lower Donchian + volume confirmation + ATR trending
            elif close_4h[i] < lower_donchian_aligned[i] and volume_confirm and atr_trending:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Close crosses back inside Donchian channel OR ATR regime shifts to ranging
            if position == 1:  # Long position
                if close_4h[i] < upper_donchian_aligned[i] or not atr_trending:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_4h[i] > lower_donchian_aligned[i] or not atr_trending:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals