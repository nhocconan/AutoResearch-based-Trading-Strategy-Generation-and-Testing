#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout + 1d volume regime + 1w trend filter
# - Primary signal: 6h price breaks above Camarilla R4 (bullish) or below S4 (bearish) from prior 1d
# - Volume confirmation: 6h volume > 1.5x 20-period median volume (ensures participation)
# - Trend filter: 1w EMA200 - price must be above EMA for longs, below for shorts (higher timeframe alignment)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Camarilla breakouts capture strong moves, volume filter avoids false breakouts, 1w EMA ensures alignment with major trend

name = "6h_1d_1w_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels (based on prior 1d OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels calculation
    # R4 = close + ((high - low) * 1.1/2)
    # S4 = close - ((high - low) * 1.1/2)
    camarilla_r4 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    camarilla_s4 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (use prior completed 1d bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Pre-compute 6h volume regime: volume > 1.5x 20-period median volume
    volume = prices['volume'].values
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > (1.5 * median_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(ema_200_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        close_price = prices['close'].iloc[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Camarilla R3 (profit taking) OR below 1w EMA200 (trend change)
            # Camarilla R3 = close + ((high - low) * 1.1/4)
            camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
            camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
            if (close_price < camarilla_r3_aligned[i] or 
                close_price < ema_200_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Camarilla S3 (profit taking) OR above 1w EMA200 (trend change)
            # Camarilla S3 = close - ((high - low) * 1.1/4)
            camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
            camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
            if (close_price > camarilla_s3_aligned[i] or 
                close_price > ema_200_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume confirmation and 1w EMA200 filter
            # Long: price breaks above Camarilla R4 AND volume regime AND price above 1w EMA200
            if (close_price > camarilla_r4_aligned[i] and 
                volume_regime[i] and 
                close_price > ema_200_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Camarilla S4 AND volume regime AND price below 1w EMA200
            elif (close_price < camarilla_s4_aligned[i] and 
                  volume_regime[i] and 
                  close_price < ema_200_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals