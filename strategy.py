#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h 55-period EMA trend + 1d Williams %R(14) mean reversion + volume confirmation.
# Long when EMA trending up (EMA55 rising) and Williams %R oversold (< -80) with volume spike.
# Short when EMA trending down (EMA55 falling) and Williams %R overbought (> -20) with volume spike.
# Williams %R provides mean-reversion signals within the trend, avoiding counter-trend trades.
# Volume spike (>1.5x 20-period average) confirms momentum. Target: 20-40 trades/year.
# Works in bull/bear: EMA filter ensures trend alignment, Williams %R avoids chasing extremes.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Highest High and Lowest Low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_1d) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Align Williams %R to 6h timeframe (needs 2-bar delay for confirmation)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r, additional_delay_bars=2)
    
    # Calculate 60-period EMA on 6h data (trend filter)
    close = prices['close'].values
    ema60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # EMA slope (trend direction) - positive = rising, negative = falling
    ema_slope = np.diff(ema60, prepend=ema60[0])
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if np.isnan(ema60[i]) or np.isnan(ema_slope[i]) or np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Williams %R levels for mean reversion
        williams_oversold = williams_r_aligned[i] < -80
        williams_overbought = williams_r_aligned[i] > -20
        
        if position == 0:
            if volume_confirm:
                # Long: EMA rising + Williams oversold (mean reversion long in uptrend)
                if ema_slope[i] > 0 and williams_oversold:
                    signals[i] = 0.25
                    position = 1
                # Short: EMA falling + Williams overbought (mean reversion short in downtrend)
                elif ema_slope[i] < 0 and williams_overbought:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if EMA turns down or Williams becomes overbought (take profit)
                if ema_slope[i] < 0 or williams_r_aligned[i] > -20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if EMA turns up or Williams becomes oversold (take profit)
                if ema_slope[i] > 0 or williams_r_aligned[i] < -80:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_EMA60_Trend_1dWilliamsR_MeanRev_Volume"
timeframe = "6h"
leverage = 1.0