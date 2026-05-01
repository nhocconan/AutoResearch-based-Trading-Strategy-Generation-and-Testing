#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
# Long when price > Alligator Jaw (13-period SMMA) AND 1d EMA50 uptrend AND volume > 1.5x 20-period volume median.
# Short when price < Alligator Lips (8-period SMMA) AND 1d EMA50 downtrend AND volume > 1.5x 20-period volume median.
# Uses ATR(14) stoploss: exit long if price < highest_since_entry - 2.5*ATR(14), exit short if price > lowest_since_entry + 2.5*ATR(14).
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year on 12h timeframe.
# Williams Alligator identifies trend absence/presence via three smoothed moving averages (Jaw, Teeth, Lips).
# In ranging markets, the lines intertwine; in trends, they diverge with Jaw (slowest) on one side, Lips (fastest) on the other.
# This provides a strong trend filter that works in both bull and bear markets by identifying when a trend is established.
# Volume confirmation ensures breakouts have participation, reducing false signals.
# ATR stoploss adapts to volatility while respecting engine semantics (close-based exit).

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_ATR_v1"
timeframe = "12h"
leverage = 1.0

def smma(values, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=np.float64)
    result = np.full_like(values, np.nan, dtype=np.float64)
    # First value is simple SMA
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CURRENT_VALUE) / N
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA of median price, shifted 8 bars forward
    # Teeth: 8-period SMMA of median price, shifted 5 bars forward  
    # Lips: 5-period SMMA of median price, shifted 3 bars forward
    median_price = (high + low) / 2.0
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply forward shifts (Alligator specific)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Invalidate the shifted values at the beginning
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Calculate 14-period ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for Alligator, EMA, volume, and ATR
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(vol_median_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Price > Jaw AND uptrend AND volume spike
            if curr_close > jaw[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Price < Lips AND downtrend AND volume spike
            elif curr_close < lips[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR break below Teeth (reversal) OR trend reversal
            stop_price = highest_since_entry - 2.5 * curr_atr
            if curr_close < stop_price or curr_close < teeth[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR break above Teeth (reversal) OR trend reversal
            stop_price = lowest_since_entry + 2.5 * curr_atr
            if curr_close > stop_price or curr_close > teeth[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals