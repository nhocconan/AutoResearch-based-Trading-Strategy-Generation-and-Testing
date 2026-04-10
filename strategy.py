#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w volume spike and 1w ADX trend filter
# - Entry: Long when Williams %R < -80 (oversold) + 1w volume > 1.8x 20-period average + 1w ADX > 20
#          Short when Williams %R > -20 (overbought) + 1w volume > 1.8x 20-period average + 1w ADX > 20
# - Exit: Close-based reversal - exit long when Williams %R > -50, exit short when Williams %R < -50
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses Williams %R from daily data for mean reversion signals, weekly volume for confirmation, weekly ADX for trend filter
# - Target: 7-25 trades/year (30-100 total over 4 years) to stay within HARD MAX: 150 total
# - Designed for 1d timeframe with strict volume confirmation (1.8x) and moderate trend filter (ADX>20) to capture reversals in trending markets
# - Weekly timeframe provides reliable trend/volume confirmation for daily mean reversion trading
# - Williams %R is effective in both bull and bear markets as it identifies extreme price levels

name = "1d_1w_williamsr_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC for Williams %R
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Pre-compute 1w volume for confirmation
    volume_1w = df_1w['volume'].values
    
    # Pre-compute 1w OHLC for ADX calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r)
    
    # Calculate 1w volume moving average (20-period)
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w ADX (14-period)
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1w = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    minus_di_1w = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = pd.Series(dx_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all HTF data to 1d timeframe
    williams_r_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close_1d}), williams_r)  # Dummy df for alignment
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Also need to align raw 1w volume for confirmation check
    volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_1w_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current Williams %R value
        wr_value = williams_r_aligned[i]
        
        # Volume confirmation: current 1w volume > 1.8x 20-period average
        volume_confirmation = volume_1w_aligned[i] > 1.8 * volume_ma_aligned[i]
        
        # Trend filter: 1w ADX > 20 indicates sufficient trending market
        trend_filter = adx_1w_aligned[i] > 20.0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) + volume confirmation + trend filter
            if (wr_value < -80.0 and 
                volume_confirmation and 
                trend_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R > -20 (overbought) + volume confirmation + trend filter
            elif (wr_value > -20.0 and 
                  volume_confirmation and 
                  trend_filter):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when Williams %R > -50 (recovering from oversold)
            # Exit short when Williams %R < -50 (declining from overbought)
            if position == 1:
                if wr_value > -50.0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if wr_value < -50.0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals