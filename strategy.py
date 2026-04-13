#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
    # Long: Alligator jaws < teeth < lips (bullish alignment) AND price > lips AND volume > 1.5x 20-period average AND price > 1d EMA50
    # Short: Alligator jaws > teeth > lips (bearish alignment) AND price < lips AND volume > 1.5x 20-period average AND price < 1d EMA50
    # Exit: Alligator alignment breaks (jaws-teeth-lips not in proper order) OR price crosses midline (teeth)
    # Using 1d for EMA50 trend filter, 12h for Alligator and volume
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 12-37 trades/year (~50-150 over 4 years) to avoid overtrading
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaw (Blue): 13-period SMMA smoothed by 8 periods
    # Teeth (Red): 8-period SMMA smoothed by 5 periods  
    # Lips (Green): 5-period SMMA smoothed by 3 periods
    def smma(source, period):
        """Smoothed Moving Average"""
        result = np.full_like(source, np.nan)
        if len(source) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    # Calculate SMMA components
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply smoothing periods
    jaw = smma(jaw_raw, 8)
    teeth = smma(teeth_raw, 5)
    lips = smma(lips_raw, 3)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Wait for Alligator to stabilize
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if price > 1d EMA50, only short if price < 1d EMA50
        long_trend_ok = close[i] > ema_1d_aligned[i]
        short_trend_ok = close[i] < ema_1d_aligned[i]
        
        # Alligator alignment
        bullish_alignment = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        bearish_alignment = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Price relative to lips (green line)
        price_above_lips = close[i] > lips[i]
        price_below_lips = close[i] < lips[i]
        
        # Entry logic: Alligator alignment + volume + trend
        long_entry = bullish_alignment and price_above_lips and vol_confirm and long_trend_ok
        short_entry = bearish_alignment and price_below_lips and vol_confirm and short_trend_ok
        
        # Exit logic: Alligator alignment breaks OR price crosses midline (teeth)
        long_exit = not bullish_alignment or close[i] < teeth[i]
        short_exit = not bearish_alignment or close[i] > teeth[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_williams_alligator_volume_trend_v1"
timeframe = "12h"
leverage = 1.0