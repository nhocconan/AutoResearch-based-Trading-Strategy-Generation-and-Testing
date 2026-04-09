#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with volume confirmation and weekly trend filter
# - Primary signal: 6h price breaking above R4 or below S4 Camarilla levels from prior 1d session
# - Trend filter: 1w EMA200 - price must be above EMA for longs, below for shorts (higher timeframe alignment)
# - Volume confirmation: 6h volume > 1.5x 20-period average volume (avoid low-participation breakouts)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Camarilla levels provide dynamic support/resistance, weekly EMA200 filter ensures alignment with major trend

name = "6h_1w_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Pre-compute prior day's Camarilla levels (using prior day's OHLC to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for prior day (shifted by 1 to use prior day's data)
    # Camarilla levels: based on prior day's range
    prior_high = np.roll(high_1d, 1)  # prior day's high
    prior_low = np.roll(low_1d, 1)    # prior day's low
    prior_close = np.roll(close_1d, 1) # prior day's close
    
    # First element will be invalid due to roll, set to nan
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Calculate Camarilla levels
    R4 = prior_close + ((prior_high - prior_low) * 1.1 / 2)
    R3 = prior_close + ((prior_high - prior_low) * 1.1 / 4)
    S3 = prior_close - ((prior_high - prior_low) * 1.1 / 4)
    S4 = prior_close - ((prior_high - prior_low) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # 6h volume regime: volume > 1.5x 20-period average volume
    volume = prices['volume'].values
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_regime = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_aligned[i]) or
            np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        close_price = prices['close'].iloc[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below R3 (take profit) OR price crosses below S4 (stop loss) OR weekly trend turns bearish
            if (close_price < R3_aligned[i] or 
                close_price < S4_aligned[i] or
                close_price < ema_200_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above S3 (take profit) OR price crosses above R4 (stop loss) OR weekly trend turns bullish
            if (close_price > S3_aligned[i] or 
                close_price > R4_aligned[i] or
                close_price > ema_200_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume confirmation and weekly trend filter
            # Long: price breaks above R4 with volume regime AND price above weekly EMA200
            if (close_price > R4_aligned[i] and 
                volume_regime[i] and 
                close_price > ema_200_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S4 with volume regime AND price below weekly EMA200
            elif (close_price < S4_aligned[i] and 
                  volume_regime[i] and 
                  close_price < ema_200_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals