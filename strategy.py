#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX trend filter + 1w volume regime
# - Primary signal: Williams %R(14) on 6h for overbought/oversold conditions
# - Trend filter: 1d ADX(14) > 25 to ensure we trade in trending markets only
# - Volume regime: 1w volume ratio (current/20-period average) > 1.2 to confirm participation
# - Logic: In high-volume trending markets (ADX>25, vol_ratio>1.2):
#   * Long when Williams %R crosses above -80 from below (oversold bounce)
#   * Short when Williams %R crosses below -20 from above (overbought rejection)
# - Works in bull/bear: ADX filter ensures we only trade when trend is present
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-35 trades/year (50-140 total over 4 years) per 6h strategy guidelines
# - Stoploss: exit when Williams %R returns to -50 level (mean reversion)

name = "6h_1d_1w_williams_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = np.where(tr_14 > 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 > 0, 100 * dm_minus_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 1w volume regime
    volume_1w = df_1w['volume'].values
    avg_volume_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume_1w / avg_volume_20
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1w, volume_ratio)
    
    # Pre-compute 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    
    williams_r = np.where((highest_high - lowest_low) != 0,
                          -100 * (highest_high - close_6h) / (highest_high - lowest_low),
                          -50)  # neutral when no range
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_ratio_aligned[i]) or
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R returns to -50 (mean reversion)
            if williams_r[i] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns to -50 (mean reversion)
            if williams_r[i] <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entries in high-volume trending markets
            if (adx_aligned[i] > 25 and volume_ratio_aligned[i] > 1.2):
                # Long: Williams %R crosses above -80 from below (oversold bounce)
                if williams_r[i] > -80 and williams_r[i-1] <= -80:
                    position = 1
                    signals[i] = 0.25
                # Short: Williams %R crosses below -20 from above (overbought rejection)
                elif williams_r[i] < -20 and williams_r[i-1] >= -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals