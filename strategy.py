#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d ADX trend filter
# - Uses 6h Williams %R(14) for extreme oversold/overbought conditions
# - Enters long when Williams %R < -80 (oversold) and 1d ADX < 25 (low trend/range market)
# - Enters short when Williams %R > -20 (overbought) and 1d ADX < 25
# - Exits when Williams %R returns to -50 (mean reversion target) or opposite extreme
# - Position size: 0.25 (25% of capital) to limit drawdown in volatile regimes
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years) to minimize fee drag
# - Works in bull markets (mean reversion in pullbacks) and bear markets (mean reversion in rallies)
# - Williams %R identifies exhaustion points; ADX filter avoids strong trends where mean reversion fails

name = "6h_1d_williamsr_adx_meanrev_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d True Range for ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 1d Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # 1d ADX(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / np.where(atr_1d != 0, atr_1d, 1e-10)
    minus_di_1d = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / np.where(atr_1d != 0, atr_1d, 1e-10)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / np.where((plus_di_1d + minus_di_1d) != 0, (plus_di_1d + minus_di_1d), 1e-10)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / np.where((highest_high_14 - lowest_low_14) != 0, (highest_high_14 - lowest_low_14), 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or
            adx_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R returns to -50 (mean reversion) or goes above -20 (overbought)
            if williams_r[i] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns to -50 (mean reversion) or goes below -80 (oversold)
            if williams_r[i] <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with low ADX (range/low trend market)
            if (williams_r[i] <= -80 and  # Oversold
                adx_1d_aligned[i] < 25):   # Low trend/range market
                position = 1
                signals[i] = 0.25
            elif (williams_r[i] >= -20 and   # Overbought
                  adx_1d_aligned[i] < 25):   # Low trend/range market
                position = -1
                signals[i] = -0.25
    
    return signals