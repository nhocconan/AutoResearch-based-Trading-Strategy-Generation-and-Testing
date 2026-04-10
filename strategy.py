#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume spike and 1w trend filter
# - Primary: 6h price breaks above/below Camarilla pivot levels (H4/L4) from prior 1d session
# - HTF 1d: Volume spike > 2.0x 20-period MA for confirmation (avoids low-volume breakouts)
# - HTF 1w: Close > weekly SMA50 for long bias, < weekly SMA50 for short bias (trend filter)
# - Long: Close > H4 + volume spike + weekly uptrend
# - Short: Close < L4 + volume spike + weekly downtrend
# - Exit: Close crosses back inside H4/L4 band
# - Position sizing: 0.25 (discrete level, balances return/drawdown, reduces fee churn)
# - Works in bull/bear: Camarilla adapts to volatility, volume filters false breakouts, weekly trend avoids counter-trend trades
# - Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_1d_1w_camarilla_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 25 or len(df_1w) < 55:  # Need enough data for calculations
        return np.zeros(n)
    
    # Pre-compute 6h data
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Pre-compute 1w data
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla pivot levels (H4, L4) - using prior day only
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d[i-1])):
            # Camarilla levels based on prior day's range
            range_ = high_1d[i-1] - low_1d[i-1]
            camarilla_h4[i] = close_1d[i-1] + range_ * 1.1 / 2
            camarilla_l4[i] = close_1d[i-1] - range_ * 1.1 / 2
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        if not np.isnan(volume_1d[i-19:i+1]).any():
            volume_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Calculate 1w SMA50
    sma_50_1w = np.full(len(close_1w), np.nan)
    for i in range(49, len(close_1w)):
        if not np.isnan(close_1w[i-49:i+1]).any():
            sma_50_1w[i] = np.mean(close_1w[i-49:i+1])
    
    # Align all HTF indicators to 6h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i]) or 
            np.isnan(sma_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period MA
        volume_spike = volume_1d_aligned[i] > 2.0 * volume_ma_20_1d_aligned[i]
        
        # Weekly trend filter: close > weekly SMA50 for uptrend, < for downtrend
        weekly_uptrend = close_1w_aligned[i] > sma_50_1w_aligned[i]
        weekly_downtrend = close_1w_aligned[i] < sma_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Close > H4 + volume spike + weekly uptrend
            if close_6h[i] > camarilla_h4_aligned[i] and volume_spike and weekly_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: Close < L4 + volume spike + weekly downtrend
            elif close_6h[i] < camarilla_l4_aligned[i] and volume_spike and weekly_downtrend:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Close crosses back inside H4/L4 band
            if position == 1:  # Long position
                if close_6h[i] < camarilla_h4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_6h[i] > camarilla_l4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals