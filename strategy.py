#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d/1w trend filter with volume confirmation
# - Williams %R(14) on 12h: oversold < -80 for long, overbought > -20 for short
# - Trend filter: 1d close > 1w EMA50 for long bias, 1d close < 1w EMA50 for short bias
# - Volume confirmation: 12h volume > 1.3x 20-period volume SMA
# - Exit when Williams %R reverts to midpoint (-50) or volume drops
# - Target: 20-40 trades/year to minimize fee drag while capturing mean reversions in extremes
# - Works in bull/bear: Williams %R captures overextended moves; trend filter ensures alignment with higher timeframe momentum

name = "12h_1d_1w_williamsr_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop for Williams %R and volume (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Load 1d data ONCE before loop for trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Load 1w data ONCE before loop for EMA50 trend (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 12h Williams %R (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 12h timeframe (already in 12h, but align for consistency)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Pre-compute 12h volume SMA for confirmation
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Pre-compute 1d close for trend filter
    close_1d = df_1d['close'].values
    
    # Pre-compute 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    for i in range(60, n):  # Start after 60-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_sma_20_12h_aligned[i]) or
            np.isnan(close_1d[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.3x 20-period volume SMA
        volume_12h_current = df_12h['volume'].values
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h_current)
        vol_confirm = volume_12h_aligned[i] > 1.3 * volume_sma_20_12h_aligned[i]
        
        # Williams %R conditions
        wr = williams_r_aligned[i]
        oversold = wr < -80
        overbought = wr > -20
        exit_level = abs(wr + 50) < 10  # Near midpoint (-50 ±10)
        
        # Trend filter: 1d close vs 1w EMA50
        close_1d_current = close_1d[i]
        ema_50_1w_current = ema_50_1w_aligned[i]
        bullish_trend = close_1d_current > ema_50_1w_current
        bearish_trend = close_1d_current < ema_50_1w_current
        
        # Entry conditions
        enter_long = oversold and bullish_trend and vol_confirm
        enter_short = overbought and bearish_trend and vol_confirm
        
        # Exit conditions
        exit_long = exit_level or not vol_confirm  # Exit when WR reverts or volume drops
        exit_short = exit_level or not vol_confirm
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals