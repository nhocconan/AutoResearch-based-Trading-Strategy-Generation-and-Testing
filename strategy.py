#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels from prior 4h candle
# - Trend filter: 4h EMA50 - ensures alignment with higher timeframe trend
# - Volume confirmation: 1h volume > 20-period median volume (avoid low-participation signals)
# - Session filter: Only trade 08-20 UTC to reduce noise
# - Position size: 0.20 (discrete level) to minimize fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) for 1h timeframe
# - Works in bull/bear: Camarilla pivots adapt to volatility, EMA50 filter avoids counter-trend trades

name = "1h_4h_camarilla_breakout_volume_v1"
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
    
    # Pre-compute Camarilla levels from prior 4h candle
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: H3/L3 = C ± (H-L)*1.1/2
    camarilla_h3 = close_4h + (high_4h - low_4h) * 1.1 / 2
    camarilla_l3 = close_4h - (high_4h - low_4h) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (use prior completed 4h bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Volume confirmation: 1h volume > 20-period median volume
    volume = prices['volume'].values
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_regime[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below Camarilla H3 OR below 4h EMA50
            if prices['close'].iloc[i] < camarilla_h3_aligned[i] or prices['close'].iloc[i] < ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Price crosses above Camarilla L3 OR above 4h EMA50
            if prices['close'].iloc[i] > camarilla_l3_aligned[i] or prices['close'].iloc[i] > ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and 4h EMA50 filter
            # Long: Price breaks above Camarilla H3 AND volume regime AND price above 4h EMA50
            if (prices['close'].iloc[i] > camarilla_h3_aligned[i] and 
                volume_regime[i] and 
                prices['close'].iloc[i] > ema_50_4h_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short: Price breaks below Camarilla L3 AND volume regime AND price below 4h EMA50
            elif (prices['close'].iloc[i] < camarilla_l3_aligned[i] and 
                  volume_regime[i] and 
                  prices['close'].iloc[i] < ema_50_4h_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals