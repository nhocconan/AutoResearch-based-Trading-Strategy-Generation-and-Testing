#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R(14) mean reversion with 1d EMA(50) trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) and price > 1d EMA50 (uptrend).
# Short when Williams %R > -20 (overbought) and price < 1d EMA50 (downtrend).
# Volume > 1.3x 20-period average confirms mean reversion bounce.
# Target: 20-40 trades/year by requiring oversold/overbought extremes + trend alignment + volume.
# Works in bull/bear: EMA filter ensures mean reversion trades align with higher timeframe trend,
# avoiding counter-trend trades in strong trends while capturing pullbacks in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Williams %R(14) on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(williams_r[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        # Trend filter: price relative to 1d EMA50
        price_above_ema = price > ema_50_aligned[i]
        price_below_ema = price < ema_50_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: oversold and price above EMA (bullish mean reversion)
                if williams_r[i] < -80 and price_above_ema:
                    signals[i] = 0.25
                    position = 1
                # Short: overbought and price below EMA (bearish mean reversion)
                elif williams_r[i] > -20 and price_below_ema:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Williams %R returns to overbought or price crosses below EMA
                if williams_r[i] > -20 or price < ema_50_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Williams %R returns to oversold or price crosses above EMA
                if williams_r[i] < -80 or price > ema_50_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR14_1dEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0