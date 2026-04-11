#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w trend filter and volume confirmation
# - Williams %R(14) on 1d: oversold < -80 for long, overbought > -20 for short
# - Trend filter: 1d close > 1w EMA50 for long bias, 1d close < 1w EMA50 for short bias
# - Volume confirmation: 1d volume > 1.2x 20-period volume SMA
# - Exit when Williams %R reverts to midpoint (-50) or volume drops below average
# - Target: 15-25 trades/year to minimize fee drag while capturing mean reversions in extremes
# - Works in both bull/bear: mean reversion in ranges, trend filter avoids counter-trend in strong moves

name = "1d_1w_williamsr_volume_v1"
timeframe = "1d"
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
    
    # Load 1d data ONCE before loop for Williams %R and volume (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Load 1w data ONCE before loop for EMA50 trend (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 1d timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    for i in range(60, n):  # Start after 60-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(close_1d[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.2x 20-period volume SMA
        volume_1d_current = df_1d['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        vol_confirm = volume_1d_aligned[i] > 1.2 * volume_sma_20_1d_aligned[i]
        
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