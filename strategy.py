#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions - reversals from extreme levels often work
# 1d EMA34 ensures we only take reversals in the direction of the higher timeframe trend
# Volume spike (>1.5 x 20-period EMA) confirms reversal validity with strong participation
# Discrete position sizing (0.25) controls fee drag while allowing meaningful exposure
# Target: 60-120 total trades over 4 years (15-30/year) for optimal risk-adjusted returns
# Works in bull markets by catching pullbacks in uptrends, works in bear by catching bounces in downtrends
# Focus on BTC/ETH as primary symbols with SOL as secondary confirmation

name = "6h_WilliamsR_Reversal_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation (volume spike > 1.5 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need enough data for EMA34 calculation
        return np.zeros(n)
    
    # 1d EMA34 calculation
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Williams %R calculation)
    start_idx = 35
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_confirmation[i]) or np.isnan(close_1d[-1]) if len(close_1d) > 0 else False):
            signals[i] = 0.0
            continue
        
        # Current 1d close for trend comparison (use last available 1d close)
        idx_1d = min(len(close_1d) - 1, i // 24)  # 24x 6h bars in 1d, but using aligned data is safer
        # Instead, we'll use the aligned EMA and compare with 1d close from the HTF data
        # Get the last 1d close that corresponds to current time
        # Since we don't have direct access, we'll use a simpler approach: trend is EMA direction
        # But for simplicity, we'll use price vs EMA as trend filter
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 (oversold reversal) with volume confirmation and price > EMA34
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and volume_confirmation[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) with volume confirmation and price < EMA34
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and volume_confirmation[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -20 (overbought) OR price crosses below EMA34 (trend change)
            if williams_r[i] < -20 and williams_r[i-1] >= -20 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -80 (oversold) OR price crosses above EMA34 (trend change)
            if williams_r[i] > -80 and williams_r[i-1] <= -80 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals