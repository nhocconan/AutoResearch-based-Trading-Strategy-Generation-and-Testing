#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d EMA200 trend filter and volume confirmation
# - Williams %R(14): measures overbought/oversold levels (-80 to -20 for mean reversion)
# - 1d EMA200 trend filter: ensures we trade with higher timeframe trend (bullish/bearish)
# - Volume confirmation: current volume > 1.5x 20-period average to avoid false signals
# - Entry: Williams %R < -80 (oversold) in bullish trend OR > -20 (overbought) in bearish trend
# - Exit: Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
# - Position size: 0.25 (25% of capital) for balanced risk/return
# - Target: 20-40 trades/year on 4h (80-160 total over 4 years) to minimize fee drag
# - Williams %R is effective in ranging markets and captures reversals in trends

name = "4h_1d_williamsr_mean_reversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Pre-compute 1d close for trend alignment
    close_1d_current = df_1d['close'].values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_current)
    
    # Pre-compute 4h Williams %R (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50.0  # Neutral when range is zero
    )
    
    # Pre-compute 4h volume average (20-period)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(williams_r[i]) or np.isnan(trend_aligned[i]) or 
            np.isnan(close_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Williams %R conditions
        oversold = williams_r[i] < -80.0
        overbought = williams_r[i] > -20.0
        exit_long = williams_r[i] > -50.0  # Exit long when crosses above -50
        exit_short = williams_r[i] < -50.0  # Exit short when crosses below -50
        
        # 1d trend filter: price > EMA200 = bullish, price < EMA200 = bearish
        bullish_trend = close_1d_aligned[i] > trend_aligned[i]
        bearish_trend = close_1d_aligned[i] < trend_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R oversold AND bullish trend AND volume confirmation
            if oversold and bullish_trend and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R overbought AND bearish trend AND volume confirmation
            elif overbought and bearish_trend and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R crosses back through -50
            exit_condition = (position == 1 and exit_long) or (position == -1 and exit_short)
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals