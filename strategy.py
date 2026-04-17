#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d EMA50 trend filter and volume confirmation.
# Uses Williams %R(14) on 12h to identify overbought/oversold conditions.
# Enters long when Williams %R crosses above -80 (oversold) with volume and price above 1d EMA50.
# Enters short when Williams %R crosses below -20 (overbought) with volume and price below 1d EMA50.
# Williams %R is effective in ranging markets (2025-2026) and captures reversals in trends.
# Designed for low turnover (target: 15-35 trades/year) with clear entry/exit rules.
# Works in bull markets (buying oversold dips) and bear markets (selling overbought rallies).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R(14) on 12h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    period = 14
    highest_high = pd.Series(high_12h).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_12h).rolling(window=period, min_periods=period).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    rr = np.where(rr == 0, 1e-10, rr)
    williams_r = ((highest_high - close_12h) / rr) * -100
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to main timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average (moderate to balance frequency)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(50, 20)  # Need sufficient data for EMA50 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Williams %R conditions
        wr = williams_r_aligned[i]
        wr_above_oversold = wr > -80  # Rising from oversold
        wr_below_overbought = wr < -20  # Falling from overbought
        
        # Trend filter: price relative to 1d EMA50
        price_above_ema = close[i] > ema50_aligned[i]
        price_below_ema = close[i] < ema50_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below with volume and above EMA50
            if i > start_idx:
                wr_prev = williams_r_aligned[i-1]
                wr_cross_above_80 = (wr_prev <= -80) and (wr > -80)
                if wr_cross_above_80 and price_above_ema and volume_filter:
                    signals[i] = 0.25
                    position = 1
            # Short: Williams %R crosses below -20 from above with volume and below EMA50
            if i > start_idx:
                wr_prev = williams_r_aligned[i-1]
                wr_cross_below_20 = (wr_prev >= -20) and (wr < -20)
                if wr_cross_below_20 and price_below_ema and volume_filter:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses below -50 OR price crosses below EMA50
            if i > start_idx:
                wr_prev = williams_r_aligned[i-1]
                wr_cross_below_50 = (wr_prev >= -50) and (wr < -50)
                price_cross_below_ema = (close[i-1] >= ema50_aligned[i-1]) and (close[i] < ema50_aligned[i])
                if wr_cross_below_50 or price_cross_below_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses above -50 OR price crosses above EMA50
            if i > start_idx:
                wr_prev = williams_r_aligned[i-1]
                wr_cross_above_50 = (wr_prev <= -50) and (wr > -50)
                price_cross_above_ema = (close[i-1] <= ema50_aligned[i-1]) and (close[i] > ema50_aligned[i])
                if wr_cross_above_50 or price_cross_above_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0