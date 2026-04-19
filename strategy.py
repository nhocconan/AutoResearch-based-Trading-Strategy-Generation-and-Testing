#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d ADX trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions (-20 to -80 range)
# ADX > 25 filters for trending markets to avoid whipsaws in ranging conditions
# Volume > 1.5x 20-period average confirms conviction
# In trending markets (ADX > 25): buy when Williams %R crosses above -80 from below, sell when crosses below -20 from above
# In ranging markets (ADX <= 25): mean reversion at Williams %R extremes (-80 oversold, -20 overbought)
# Position size: 0.25 to limit drawdown
# Target: 20-40 trades/year per symbol to stay within frequency limits

name = "12h_WilliamsR_ADX_Volume"
timeframe = "12h"
leverage = 1.0

def williams_r(high, low, close, period):
    """Williams %R indicator"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)
    return wr.values

def adx(high, low, close, period):
    """ADX indicator"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(high[i] - high[i-1], 0)
        minus_dm[i] = max(low[i-1] - low[i], 0)
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        elif plus_dm[i] < minus_dm[i]:
            plus_dm[i] = 0
        else:
            minus_dm[i] = 0
            
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilders_smoothing(tr, period)
    plus_dm14 = wilders_smoothing(plus_dm, period)
    minus_dm14 = wilders_smoothing(minus_dm, period)
    
    plus_di14 = 100 * plus_dm14 / tr14
    minus_di14 = 100 * minus_dm14 / tr14
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    dx = np.where((plus_di14 + minus_di14) == 0, 0, dx)
    adx_result = wilders_smoothing(dx, period)
    return adx_result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period)
    wr_14 = williams_r(high_1d, low_1d, close_1d, 14)
    
    # Calculate ADX (14-period)
    adx_14 = adx(high_1d, low_1d, close_1d, 14)
    
    # Align indicators to 12h timeframe
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr_14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Get 12h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure Williams %R (14*2+6), ADX (14*2+6), and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        wr_val = wr_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Trend determination
        is_trending = adx_val > 25
        is_ranging = adx_val <= 25
        
        if position == 0:
            # Determine entry based on regime
            if is_trending and volume_confirmed:
                # Trending regime: Williams %R crossovers
                if wr_val > -80 and wr_aligned[i-1] <= -80:
                    signals[i] = 0.25
                    position = 1
                elif wr_val < -20 and wr_aligned[i-1] >= -20:
                    signals[i] = -0.25
                    position = -1
            elif is_ranging and volume_confirmed:
                # Ranging regime: mean reversion at extremes
                if wr_val <= -80:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif wr_val >= -20:  # Overbought
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            if is_trending:
                # In trending regime, exit when Williams %R crosses below -50 from above
                if wr_val < -50 and wr_aligned[i-1] >= -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In ranging regime, exit when Williams %R returns from oversold
                if wr_val > -50:  # Returning from oversold
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            if is_trending:
                # In trending regime, exit when Williams %R crosses above -50 from below
                if wr_val > -50 and wr_aligned[i-1] <= -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In ranging regime, exit when Williams %R returns from overbought
                if wr_val < -50:  # Returning from overbought
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals