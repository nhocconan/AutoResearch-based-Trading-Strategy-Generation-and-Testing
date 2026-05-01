#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator (Jaw/Teeth/Lips) with 1d trend filter (close > SMA50) and volume confirmation.
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d SMA50 AND volume > 1.5x 20-bar average.
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d SMA50 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to limit drawdown. Session filter 08-20 UTC to avoid low-liquidity hours.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
# Williams Alligator identifies trend phases; alignment confirms strong trend. 1d SMA50 filters for higher-timeframe trend.
# Volume confirmation ensures breakouts have conviction. Works in bull (long when aligned above SMA) and bear (short when aligned below SMA).

name = "4h_WilliamsAlligator_1dSMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for SMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d SMA50 calculation
    close_1d = df_1d['close'].values
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    
    # 1d trend: price above/below SMA50
    price_above_sma = close > sma_50_aligned
    price_below_sma = close < sma_50_aligned
    
    # Williams Alligator on 4h data: SMAs of median price with offsets
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # Jaw: 13-period SMMA, offset 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # offset 8 bars into future
    jaw[:8] = np.nan  # first 8 values invalid due to offset
    
    # Teeth: 8-period SMMA, offset 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # offset 5 bars into future
    teeth[:5] = np.nan  # first 5 values invalid due to offset
    
    # Lips: 5-period SMMA, offset 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # offset 3 bars into future
    lips[:3] = np.nan  # first 3 values invalid due to offset
    
    # Alligator alignment: bullish when Lips > Teeth > Jaw, bearish when Lips < Teeth < Jaw
    bullish_align = (lips > teeth) & (teeth > jaw)
    bearish_align = (lips < teeth) & (teeth < jaw)
    
    # Volume confirmation: current 4h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(sma_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator alignment AND price > 1d SMA50 AND volume confirmation
            if (bullish_align[i] and 
                price_above_sma[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment AND price < 1d SMA50 AND volume confirmation
            elif (bearish_align[i] and 
                  price_below_sma[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: bearish Alligator alignment OR price < 1d SMA50 (trend change)
            if (bearish_align[i] or 
                not price_above_sma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bullish Alligator alignment OR price > 1d SMA50 (trend change)
            if (bullish_align[i] or 
                not price_below_sma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals