#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d trend filter and volume confirmation.
# Go long when Williams %R crosses above -80 (oversold) in a bullish trend (price > 1d EMA50).
# Go short when Williams %R crosses below -20 (overbought) in a bearish trend (price < 1d EMA50).
# Requires volume > 1.3x 14-period average for confirmation.
# Uses Williams %R(14) for mean reversion entries in trending markets.
# Target: 12-37 trades/year by requiring trend alignment + extreme %R + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_d = df_1d['close'].values
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(ema50_1d_aligned[i]):
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
        
        # Calculate Williams %R (14-period)
        lookback_start = max(0, i - 13)
        high_window = prices['high'].iloc[lookback_start:i+1].values
        low_window = prices['low'].iloc[lookback_start:i+1].values
        close = prices['close'].iloc[i]
        
        highest_high = np.max(high_window)
        lowest_low = np.min(low_window)
        
        # Avoid division by zero
        if highest_high == lowest_low:
            williams_r = -50  # neutral
        else:
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        # Current volume
        volume = prices['volume'].iloc[i]
        
        # Calculate 14-period volume average
        vol_lookback_start = max(0, i - 13)
        vol_window = prices['volume'].iloc[vol_lookback_start:i+1].values
        vol_ma_14 = np.mean(vol_window)
        
        # Volume confirmation: current volume > 1.3x 14-period average
        volume_confirm = volume > 1.3 * vol_ma_14
        
        # Trend filter: price vs daily EMA50
        bull_trend = price > ema50_1d_aligned[i]
        bear_trend = price < ema50_1d_aligned[i]
        
        if position == 0:
            # Enter long when Williams %R crosses above -80 (oversold) in bullish trend with volume
            if williams_r > -80 and williams_r <= -80 + 0.5 and volume_confirm and bull_trend:
                signals[i] = 0.25
                position = 1
            # Enter short when Williams %R crosses below -20 (overbought) in bearish trend with volume
            elif williams_r < -20 and williams_r >= -20 - 0.5 and volume_confirm and bear_trend:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R returns to neutral zone (-50)
            exit_signal = False
            
            if position == 1:
                # Exit long when Williams %R rises above -50
                if williams_r > -50:
                    exit_signal = True
            elif position == -1:
                # Exit short when Williams %R falls below -50
                if williams_r < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_Trend_Volume"
timeframe = "12h"
leverage = 1.0