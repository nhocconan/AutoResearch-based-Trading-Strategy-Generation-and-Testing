#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Long when Alligator jaws < teeth < lips (bullish alignment) AND close > EMA50(1w) AND volume > 1.5x 20-period average
# Short when Alligator jaws > teeth > lips (bearish alignment) AND close < EMA50(1w) AND volume > 1.5x 20-period average
# Exit when Alligator alignment breaks (jaws > teeth OR teeth > lips) OR EMA50(1w) trend flip
# Uses 12h primary timeframe with 1w HTF for trend filter to reduce whipsaw and avoid overtrading
# Williams Alligator: SMAs of median price (H+L)/2 with periods 13,8,5 and offsets 8,5,3
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Alligator identifies trend emergence; 1w EMA filter ensures higher timeframe alignment; volume confirms conviction

name = "12h_Williams_Alligator_1wEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Williams Alligator (based on median price)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator: three smoothed SMAs of median price
    median_price = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    # Jaws: SMA(13) of median price, offset 8 bars
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaws = np.roll(jaws, 8)  # shift right by 8 (offset)
    jaws[:8] = np.nan  # first 8 values invalid
    
    # Teeth: SMA(8) of median price, offset 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift right by 5 (offset)
    teeth[:5] = np.nan  # first 5 values invalid
    
    # Lips: SMA(5) of median price, offset 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift right by 3 (offset)
    lips[:3] = np.nan  # first 3 values invalid
    
    # Align Alligator lines to 12h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: volume > 1.5x 20-period average (balanced to avoid overtrading)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaws_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator bullish alignment (jaws < teeth < lips) AND close > EMA50(1w) AND volume spike
            if (jaws_aligned[i] < teeth_aligned[i] and 
                teeth_aligned[i] < lips_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator bearish alignment (jaws > teeth > lips) AND close < EMA50(1w) AND volume spike
            elif (jaws_aligned[i] > teeth_aligned[i] and 
                  teeth_aligned[i] > lips_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (jaws > teeth OR teeth > lips) OR close < EMA50(1w) (trend flip)
            if (jaws_aligned[i] > teeth_aligned[i] or 
                teeth_aligned[i] > lips_aligned[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks (jaws < teeth OR teeth < lips) OR close > EMA50(1w) (trend flip)
            if (jaws_aligned[i] < teeth_aligned[i] or 
                teeth_aligned[i] < lips_aligned[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals