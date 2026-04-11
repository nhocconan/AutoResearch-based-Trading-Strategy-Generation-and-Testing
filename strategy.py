#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Williams %R(14) from 6h: oversold < -80, overbought > -20
# - Long when %R crosses above -80 from below (mean reversion from oversold) with 1d uptrend (close > EMA50) and volume > 1.5x 20-period average
# - Short when %R crosses below -20 from above (mean reversion from overbought) with 1d downtrend (close < EMA50) and volume > 1.5x 20-period average
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 6h
# - Volume confirmation ensures we trade mean reversions with participation, reducing false signals
# - 1d EMA50 filter ensures we trade mean reversions in the direction of the higher timeframe trend, improving win rate in both bull and bear markets

name = "6h_1d_williamsr_meanreversion_trendfilter_v1"
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
    
    # Load 1d data ONCE before loop for trend and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Pre-compute 1d volume SMA (20-period)
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Williams %R conditions
        wr_current = williams_r[i]
        wr_previous = williams_r[i-1]
        
        # Cross above -80 from below (oversold mean reversion long)
        cross_above_80 = (wr_previous < -80) and (wr_current >= -80)
        
        # Cross below -20 from above (overbought mean reversion short)
        cross_below_20 = (wr_previous > -20) and (wr_current <= -20)
        
        # Volume confirmation: current volume > 1.5x 20-period average (using 1d aligned volume)
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # 1d trend filter: EMA50 direction
        uptrend = price_close > ema50_1d_aligned[i]
        downtrend = price_close < ema50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R cross above -80 + 1d uptrend + volume confirmation
        if cross_above_80 and uptrend and vol_confirm:
            enter_long = True
        
        # Short: Williams %R cross below -20 + 1d downtrend + volume confirmation
        if cross_below_20 and downtrend and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Williams %R cross or loss of volume confirmation
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R crosses below -20 from above (overbought) OR loss of volume confirmation
            exit_long = cross_below_20 or (not vol_confirm)
        elif position == -1:
            # Exit short if Williams %R crosses above -80 from below (oversold) OR loss of volume confirmation
            exit_short = cross_above_80 or (not vol_confirm)
        
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