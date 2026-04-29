#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 Trend + Volume Spike
# Long when Alligator is bullish (jaw < teeth < lips) AND price > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short when Alligator is bearish (jaw > teeth > lips) AND price < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit when Alligator becomes neutral (teeth between jaw and lips) or opposite signal appears
# Uses Williams Alligator (SMAs: jaw=13, teeth=8, lips=5) to identify trending vs ranging markets
# 1d EMA34 filters for higher timeframe trend alignment
# Volume confirmation ensures breakout strength
# Discrete position sizing (0.25) to control fee drag. Target: 12-25 trades/year on 6h timeframe.

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 6h data
    # Jaw: 13-period SMMA (smoothed moving average) of median price
    # Teeth: 8-period SMMA of median price
    # Lips: 5-period SMMA of median price
    median_price = (high + low) / 2.0
    
    # SMMA calculation: first value is SMA, then smoothed
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Align Alligator lines to 6h timeframe (they're already on 6h, but ensure alignment)
    jaw_aligned = align_htf_to_ltf(prices, prices[['open','high','low','close','volume']].iloc[:len(jaw)], jaw)
    teeth_aligned = align_htf_to_ltf(prices, prices[['open','high','low','close','volume']].iloc[:len(teeth)], teeth)
    lips_aligned = align_htf_to_ltf(prices, prices[['open','high','low','close','volume']].iloc[:len(lips)], lips)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # EMA34 and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema34 = ema_34_1d_aligned[i]
        curr_jaw = jaw_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_lips = lips_aligned[i]
        curr_close = close[i]
        
        # Determine Alligator state
        # Bullish: jaw < teeth < lips (all lines ascending, price above lips)
        bullish = (curr_jaw < curr_teeth) and (curr_teeth < curr_lips) and (curr_close > curr_lips)
        # Bearish: jaw > teeth > lips (all lines descending, price below lips)
        bearish = (curr_jaw > curr_teeth) and (curr_teeth > curr_lips) and (curr_close < curr_lips)
        # Neutral: otherwise (intertwined or sideways)
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Alligator becomes neutral or bearish
            if not bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator becomes neutral or bullish
            if not bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Alligator is bullish AND price > 1d EMA34 AND volume confirmation
            if bullish and curr_close > curr_ema34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Alligator is bearish AND price < 1d EMA34 AND volume confirmation
            elif bearish and curr_close < curr_ema34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals