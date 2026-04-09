#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and 1d volume confirmation
# - Primary signal: 1h price breaks above/below Camarilla H3/L3 levels from prior 4h bar
# - Trend filter: 4h EMA50 - price must be above EMA for longs, below for shorts
# - Volume confirmation: 1d volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.20 (discrete level) to minimize fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) per 1h strategy guidelines
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Works in bull/bear: Camarilla provides adaptive support/resistance, EMA50 filter ensures alignment with higher timeframe trend

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute 1d volume regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    median_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    volume_regime_1d = volume_1d > median_volume_20
    volume_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_regime_1d)
    
    # Pre-compute Camarilla levels from prior 4h bar (H3, L3)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    camarilla_h3 = close_4h + (1.1 * (high_4h - low_4h) / 6)
    camarilla_l3 = close_4h - (1.1 * (high_4h - low_4h) / 6)
    
    # Align Camarilla levels to 1h timeframe (completed 4h bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # 1h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_regime_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla L3 OR price crosses below 4h EMA50
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla H3 OR price crosses above 4h EMA50
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and 4h EMA50 filter
            # Long: price breaks above Camarilla H3 AND volume regime AND price above 4h EMA50
            if (close[i] > camarilla_h3_aligned[i] and 
                volume_regime_1d_aligned[i] and 
                close[i] > ema_50_4h_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short: price breaks below Camarilla L3 AND volume regime AND price below 4h EMA50
            elif (close[i] < camarilla_l3_aligned[i] and 
                  volume_regime_1d_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals