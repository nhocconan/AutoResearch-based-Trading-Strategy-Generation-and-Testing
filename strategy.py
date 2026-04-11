#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion + weekly trend filter + volume confirmation
# - Williams %R(14) from 6h: extreme readings (< -80 for long, > -20 for short) indicate oversold/overbought
# - Weekly trend filter: only trade in direction of weekly EMA(21) to avoid counter-trend trades
# - Volume confirmation: current 6h volume > 1.5x 20-period average to ensure breakout conviction
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Williams %R works well in ranging markets (2025+ bear/range) while weekly EMA filter avoids major trend mistakes

name = "6h_1d_williamsr_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return signals
    
    # Pre-compute weekly EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Pre-compute 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Pre-compute 6h volume SMA(20) for confirmation
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_sma_20[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Williams %R conditions
        oversold = williams_r[i] < -80  # Extreme oversold
        overbought = williams_r[i] > -20  # Extreme overbought
        
        # Weekly trend filter: price above/below weekly EMA
        price_above_weekly_ema = price_close > ema_21_1w_aligned[i]
        price_below_weekly_ema = price_close < ema_21_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R oversold + price above weekly EMA + volume confirmation
        if oversold and price_above_weekly_ema and vol_confirm:
            enter_long = True
        
        # Short: Williams %R overbought + price below weekly EMA + volume confirmation
        if overbought and price_below_weekly_ema and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Williams %R extreme or loss of volume confirmation
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R becomes overbought OR volume confirmation lost
            exit_long = (williams_r[i] > -20) or (not vol_confirm)
        elif position == -1:
            # Exit short if Williams %R becomes oversold OR volume confirmation lost
            exit_short = (williams_r[i] < -80) or (not vol_confirm)
        
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