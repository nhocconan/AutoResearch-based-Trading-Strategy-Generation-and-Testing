#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; in strong trends, these can signal continuations rather than reversals.
# Combined with 1d EMA trend filter and volume spike, this captures momentum bursts in trending markets.
# Works in both bull and bear markets by following the dominant trend on higher timeframe.
# Target: 20-40 trades/year by requiring Williams %R extreme, volume confirmation, and trend alignment.
# Entry: Long when Williams %R crosses above -20 from below, with volume spike and price > 1d EMA50.
# Short when Williams %R crosses below -80 from above, with volume spike and price < 1d EMA50.
# Exit: Opposite Williams %R cross or trend reversal.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for EMA and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on daily timeframe
    close_d = df_1d['close'].values
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R (14-period) on daily timeframe
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d_wr = df_1d['close'].values
    highest_high = pd.Series(high_d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_d).rolling(window=14, min_periods=14).min().values
    wr = -100 * (highest_high - close_d_wr) / (highest_high - lowest_low)
    # Handle division by zero
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)
    
    # Align daily data to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_d)
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr)
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol = prices['volume'].values
    vol_ma_20 = pd.Series(vol).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(wr_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price_close = prices['close'].iloc[i]
        vol_current = vol[i]
        
        # Williams %R values
        wr_current = wr_aligned[i]
        wr_prev = wr_aligned[i-1] if i > 0 else wr_current
        
        # Trend filter: price relative to daily EMA50
        above_ema = price_close > ema50_1d_aligned[i]
        below_ema = price_close < ema50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol_current > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Enter long when Williams %R crosses above -20 from below, with volume and trend
            if (wr_current > -20 and wr_prev <= -20 and volume_confirm and above_ema):
                signals[i] = 0.25
                position = 1
            # Enter short when Williams %R crosses below -80 from above, with volume and trend
            elif (wr_current < -80 and wr_prev >= -80 and volume_confirm and below_ema):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses below -80 or trend turns bearish
                if wr_current < -80 and wr_prev >= -80:
                    exit_signal = True
                elif price_close < ema50_1d_aligned[i]:  # trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R crosses above -20 or trend turns bullish
                if wr_current > -20 and wr_prev <= -20:
                    exit_signal = True
                elif price_close > ema50_1d_aligned[i]:  # trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0