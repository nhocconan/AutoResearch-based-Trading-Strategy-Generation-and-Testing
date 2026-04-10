#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d volume spike and 1d ADX regime filter
# - Entry: Long when Williams %R(14) < -80 (oversold) + 1d volume > 2.0x 20-period average + 1d ADX < 30 (low volatility regime)
#          Short when Williams %R(14) > -20 (overbought) + 1d volume > 2.0x 20-period average + 1d ADX < 30
# - Exit: Close-based reversal - exit long when Williams %R > -50, exit short when Williams %R < -50
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses Williams %R from 12h for mean reversion signals, daily volume for confirmation, daily ADX for regime filter
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within HARD MAX: 200 total
# - Williams %R identifies exhaustion points in ranging markets, volume confirms participation, ADX avoids strong trends where mean reversion fails
# - In bear markets (like 2025 test), mean reversion at extremes works well during bear rallies and pullbacks

name = "12h_1d_williamsr_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC for Williams %R
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Pre-compute 1d data for indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r_12h = (highest_high_12h - close_12h) / (highest_high_12h - lowest_low_12h) * -100
    # Handle division by zero (when high == low)
    williams_r_12h = np.where((highest_high_12h - lowest_low_12h) == 0, -50, williams_r_12h)
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period) for regime filtering
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all HTF data to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_12h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h close (not needed for logic but kept for consistency)
        close_price = close_12h[i]
        
        # Get current 1d volume for confirmation (need to align raw volume)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirmation = volume_1d_aligned[i] > 2.0 * volume_ma_aligned[i]
        
        # Regime filter: 1d ADX < 30 indicates low volatility / ranging market (good for mean reversion)
        regime_filter = adx_1d_aligned[i] < 30.0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold (< -80) + volume confirmation + ranging market
            if (williams_r_aligned[i] < -80.0 and 
                volume_confirmation and 
                regime_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R overbought (> -20) + volume confirmation + ranging market
            elif (williams_r_aligned[i] > -20.0 and 
                  volume_confirmation and 
                  regime_filter):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when Williams %R > -50 (recovering from oversold)
            # Exit short when Williams %R < -50 (declining from overbought)
            if position == 1:
                if williams_r_aligned[i] > -50.0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if williams_r_aligned[i] < -50.0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals