#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator + 1d volume confirmation + chop regime filter
    # Long: Alligator bullish alignment (jaw < teeth < lips) + volume > 1.5x 20-period avg + CHOP > 50 (range)
    # Short: Alligator bearish alignment (jaw > teeth > lips) + volume > 1.5x 20-period avg + CHOP > 50 (range)
    # Uses Williams Alligator (SMMA) for trend identification in choppy markets
    # Target: 15-35 trades/year to stay within 12h optimal range
    # Works in both bull/bear by trading range reversals with volume confirmation
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator calculation and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    median_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    # Williams Alligator: three smoothed moving averages (SMMA)
    # Jaw: SMMA(13, 8) - blue line
    # Teeth: SMMA(8, 5) - red line  
    # Lips: SMMA(5, 3) - green line
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA(i) = (SMMA(i-1) * (period-1) + data[i]) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_1d, 13)
    teeth = smma(median_1d, 8)
    lips = smma(median_1d, 5)
    
    # Calculate 1d CHOP (choppiness index) for regime filter
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    tr_1d = true_range(df_1d['high'].values, df_1d['low'].values, np.roll(df_1d['close'].values, 1))
    tr_1d[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # First TR
    
    atr_1d = np.zeros_like(tr_1d)
    for i in range(1, len(tr_1d)):
        if i < 14:
            atr_1d[i] = np.mean(tr_1d[:i+1])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14  # Wilder's smoothing
    
    # CHOP = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    chop_1d = np.full_like(df_1d['close'].values, np.nan)
    for i in range(13, len(df_1d)):
        sum_atr = np.sum(atr_1d[i-13:i+1])
        max_high = np.max(df_1d['high'].values[i-13:i+1])
        min_low = np.min(df_1d['low'].values[i-13:i+1])
        if max_high > min_low:
            chop_1d[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Calculate ATR using true range approximation for 12h timeframe
    atr_12h = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_12h[i] = tr  # Simple average for warmup
        else:
            atr_12h[i] = 0.93 * atr_12h[i-1] + 0.07 * tr  # Wilder's smoothing
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        vol_avg_20_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_avg_20_12h[i]):
            signals[i] = 0.0
            continue
        volume_confirmed = volume[i] > 1.5 * vol_avg_20_12h[i]
        
        # Regime filter: CHOP > 50 indicates ranging market (good for mean reversion)
        ranging_market = chop_aligned[i] > 50
        
        # Alligator signals: bullish (jaw < teeth < lips) or bearish (jaw > teeth > lips)
        alligator_bullish = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
        alligator_bearish = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        # Entry conditions: Alligator alignment + volume confirmation + ranging market
        entry_long = alligator_bullish and volume_confirmed and ranging_market
        entry_short = alligator_bearish and volume_confirmed and ranging_market
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_12h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_12h[i]
        
        # Execute signals
        if entry_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif entry_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "12h_1d_alligator_volume_chop_v1"
timeframe = "12h"
leverage = 1.0