#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume confirmation + 1w trend filter
# - Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs with future shift
# - Long when: Lips > Teeth > Jaw (bullish alignment) AND 1d volume > 1.5x 20-period volume SMA AND 1w close > 1w EMA50
# - Short when: Lips < Teeth < Jaw (bearish alignment) AND 1d volume > 1.5x 20-period volume SMA AND 1w close < 1w EMA50
# - Exit: Alligator lines cross (Lips-Teeth or Teeth-Jaw crossover) or loss of volume confirmation
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 12h timeframe to stay within fee drag limits
# - Uses Williams Alligator from 12h for trend, 1d for volume confirmation, 1w for trend filter

name = "12h_1d_1w_alligator_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 12h Williams Alligator
    # Jaw: 13-period SMMA, smoothed by 8 periods
    # Teeth: 8-period SMMA, smoothed by 5 periods  
    # Lips: 5-period SMMA, smoothed by 3 periods
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan, dtype=np.float64)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply forward shift as per Alligator specification
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Align 12h Alligator lines to 12h timeframe (no shift needed as we already shifted)
    jaw_aligned = jaw  # Already aligned to current bar
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w close for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    for i in range(100, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(close_1w_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        # Need to get 1d volume for current 12h bar
        # Since 12h = 2 * 12h bars per day, we use i//2 for 1d index
        vol_1d_idx = min(i // 2, len(volume_1d) - 1)
        vol_confirm = volume_1d[vol_1d_idx] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: 1w close vs 1w EMA50
        trend_bullish = close_1w_aligned[i] > ema_50_1w_aligned[i]
        trend_bearish = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Alligator signals
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Exit conditions: Alligator lines cross or loss of volume confirmation
        lips_teeth_cross = (lips_aligned[i] - teeth_aligned[i]) * (lips_aligned[max(i-1,0)] - teeth_aligned[max(i-1,0)]) <= 0
        teeth_jaw_cross = (teeth_aligned[i] - jaw_aligned[i]) * (teeth_aligned[max(i-1,0)] - jaw_aligned[max(i-1,0)]) <= 0
        exit_signal = lips_teeth_cross or teeth_jaw_cross or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if bullish_alignment and trend_bullish and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif bearish_alignment and trend_bearish and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals