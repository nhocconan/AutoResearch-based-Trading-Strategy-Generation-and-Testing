#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot breakout with 1d volume regime filter
# - Long when price breaks above H3 Camarilla pivot level (4h) AND 1d volume > 1.2x 20-period average volume
# - Short when price breaks below L3 Camarilla pivot level (4h) AND 1d volume > 1.2x 20-period average volume
# - Exit when price returns to the 4h Pivot point (PP) level
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Uses discrete position sizing 0.20 to limit fee churn
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Camarilla pivots provide precise intraday support/resistance levels
# - Volume filter ensures breakouts occur with participation
# - Session filter reduces noise during off-hours

name = "1h_4h_1d_camarilla_pivot_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 5 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Camarilla pivots (using previous 4h bar's OHLC)
    # Camarilla formulas:
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.1*(high-low)
    # H2 = close + 0.55*(high-low)
    # H1 = close + 0.275*(high-low)
    # PP = (high+low+close)/3
    # L1 = close - 0.275*(high-low)
    # L2 = close - 0.55*(high-low)
    # L3 = close - 1.1*(high-low)
    # L4 = close - 1.5*(high-low)
    
    h4 = df_4h['close'] + 1.5 * (df_4h['high'] - df_4h['low'])
    h3 = df_4h['close'] + 1.1 * (df_4h['high'] - df_4h['low'])
    h2 = df_4h['close'] + 0.55 * (df_4h['high'] - df_4h['low'])
    h1 = df_4h['close'] + 0.275 * (df_4h['high'] - df_4h['low'])
    pp = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    l1 = df_4h['close'] - 0.275 * (df_4h['high'] - df_4h['low'])
    l2 = df_4h['close'] - 0.55 * (df_4h['high'] - df_4h['low'])
    l3 = df_4h['close'] - 1.1 * (df_4h['high'] - df_4h['low'])
    l4 = df_4h['close'] - 1.5 * (df_4h['high'] - df_4h['low'])
    
    # Align 4h Camarilla levels to 1h timeframe (wait for 4h bar to close)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3.values)
    pp_aligned = align_htf_to_ltf(prices, df_4h, pp.values)
    
    # Pre-compute 1h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.2 * vol_ma)
    
    # Pre-compute 1d volume regime (to avoid low-volume days)
    vol_1d_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_1d_current = df_1d['volume'].values
    high_vol_regime = vol_1d_current > vol_1d_ma
    
    # Align 1d volume regime to 1h timeframe
    high_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(high_vol_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume filter AND high volume regime
            if (close[i] > h3_aligned[i] and 
                volume_filter[i] and 
                high_vol_regime_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short conditions: price breaks below L3 AND volume filter AND high volume regime
            elif (close[i] < l3_aligned[i] and 
                  volume_filter[i] and 
                  high_vol_regime_aligned[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to the Pivot point (PP) level
            exit_long = (position == 1 and close[i] < pp_aligned[i])
            exit_short = (position == -1 and close[i] > pp_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals