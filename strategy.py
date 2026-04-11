#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume spike confirmation
# - Williams %R(14) from 6h: long when < -80 (oversold), short when > -20 (overbought)
# - 12h EMA(50) trend filter: only long when price > EMA50, short when price < EMA50
# - Volume spike confirmation: current 6h volume > 1.5x 20-period average (using 12h aligned volume SMA)
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Williams %R works in ranging markets (mean reversion) and with trend filter avoids fighting major trends
# - Volume spike confirms momentum behind the reversal, reducing false signals

name = "6h_12h_williamsr_volume_trend_v1"
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
    
    # Load 12h data ONCE before loop for EMA trend and volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Pre-compute 12h volume SMA (20-period) for volume confirmation
    volume_12h = df_12h['volume'].values
    volume_series = pd.Series(volume_12h)
    volume_sma_20_12h = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Pre-compute 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_sma_20_aligned[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        wr_current = williams_r[i]
        
        # Williams %R conditions
        wr_oversold = wr_current < -80  # Oversold condition for long
        wr_overbought = wr_current > -20  # Overbought condition for short
        
        # Trend filter: price relative to 12h EMA50
        uptrend = price_close > ema_50_aligned[i]
        downtrend = price_close < ema_50_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average (using 12h aligned volume SMA)
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R oversold + uptrend + volume confirmation
        if wr_oversold and uptrend and vol_confirm:
            enter_long = True
        
        # Short: Williams %R overbought + downtrend + volume confirmation
        if wr_overbought and downtrend and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Williams %R level or trend change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R rises above -50 (leaving oversold) OR trend turns down
            exit_long = (wr_current > -50) or (not uptrend)
        elif position == -1:
            # Exit short if Williams %R falls below -50 (leaving overbought) OR trend turns up
            exit_short = (wr_current < -50) or (not downtrend)
        
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