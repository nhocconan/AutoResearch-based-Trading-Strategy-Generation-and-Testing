#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion + 1d trend filter + volume spike
# - Williams %R(14) on 6h: long when %R < -80 (oversold), short when %R > -20 (overbought)
# - Trend filter: 1d EMA50 > EMA200 for long bias, EMA50 < EMA200 for short bias
# - Volume confirmation: 6h volume > 2.0x 20-period average to filter weak signals
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Williams %R captures overextended moves that revert in ranging/choppy markets
# - 1d EMA filter ensures we trade with the higher timeframe trend for better win rate
# - Volume spike confirms institutional participation in the reversal
# - Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets

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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Pre-compute 1d trend bias (1 for uptrend, -1 for downtrend, 0 for neutral)
    trend_bias = np.zeros(len(ema_50_aligned))
    trend_bias[ema_50_aligned > ema_200_aligned] = 1
    trend_bias[ema_50_aligned < ema_200_aligned] = -1
    
    # Pre-compute 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(trend_bias[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Trend filter from 1d
        trend_up = trend_bias[i] == 1
        trend_down = trend_bias[i] == -1
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R oversold + uptrend + volume confirmation
        if oversold and trend_up and vol_confirm:
            enter_long = True
        
        # Short: Williams %R overbought + downtrend + volume confirmation
        if overbought and trend_down and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Williams %R level or trend reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R rises above -50 OR trend turns down
            exit_long = (williams_r[i] > -50) or (not trend_up)
        elif position == -1:
            # Exit short if Williams %R falls below -50 OR trend turns up
            exit_short = (williams_r[i] < -50) or (not trend_down)
        
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