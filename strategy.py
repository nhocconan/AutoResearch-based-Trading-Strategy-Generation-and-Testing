#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w/1d HTF trend filter and volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > lips AND 1w close > 1d EMA50 AND volume > 1.5x 20-bar average.
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < lips AND 1w close < 1d EMA50 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Williams Alligator identifies trend via smoothed medians, HTF filters ensure alignment with higher timeframe momentum, volume confirms strength.
# Primary timeframe: 12h, HTF: 1w for EMA trend and 1d for additional confirmation.

name = "12h_Williams_Alligator_1wEMA50_1dTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w and 1d data ONCE before loop for HTF indicators
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator: 3 smoothed medians (Jaw, Teeth, Lips)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2.0
    
    def smma(values, period):
        """Smoothed Moving Average"""
        result = np.full_like(values, np.nan, dtype=np.float64)
        if len(values) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply shifts: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Invalidate the rolled values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d close > 1d EMA34 for additional trend confirmation (optional)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current 12h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for Alligator and indicators
    
    for i in range(start_idx, n):
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or \
           np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_lips = lips_aligned[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume confirmation threshold
        
        # Alligator alignment signals
        bullish_alignment = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
        bearish_alignment = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        # Trend filters
        bullish_1w = curr_close > ema_50_1w_aligned[i]
        bearish_1w = curr_close < ema_50_1w_aligned[i]
        
        # Additional 1d trend confirmation
        bullish_1d = curr_close > ema_34_1d_aligned[i]
        bearish_1d = curr_close < ema_34_1d_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator alignment AND price > lips AND 1w bullish trend AND 1d bullish trend AND volume confirmation
            if (bullish_alignment and 
                curr_close > curr_lips and 
                bullish_1w and 
                bullish_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment AND price < lips AND 1w bearish trend AND 1d bearish trend AND volume confirmation
            elif (bearish_alignment and 
                  curr_close < curr_lips and 
                  bearish_1w and 
                  bearish_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: bearish Alligator alignment OR price < lips OR trend turns bearish
            if (bearish_alignment or 
                curr_close < curr_lips or 
                bearish_1w or 
                bearish_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bullish Alligator alignment OR price > lips OR trend turns bullish
            if (bullish_alignment or 
                curr_close > curr_lips or 
                bullish_1w or 
                bullish_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals