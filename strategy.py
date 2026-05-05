#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Long when Alligator jaws (13-period smoothed median) teeth (8-period) lips (5-period) are aligned bullish (jaws < teeth < lips) AND close > EMA34(1d) AND volume > 2.0x 20-period average
# Short when jaws > teeth > lips (bearish alignment) AND close < EMA34(1d) AND volume > 2.0x 20-period average
# Exit when Alligator alignment breaks (jaws-teeth-lips not in bullish/bearish order) OR close crosses EMA34(1d) in opposite direction
# Uses 12h primary timeframe with 1d HTF for trend filter to capture medium-term moves in both bull and bear markets
# Williams Alligator identifies trending vs ranging markets; volume confirmation filters weak breakouts
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "12h_Williams_Alligator_1dEMA34_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaws: 13-period SMMA of median price, smoothed 8 bars ahead
    # Teeth: 8-period SMMA of median price, smoothed 5 bars ahead
    # Lips: 5-period SMMA of median price, smoothed 3 bars ahead
    median_price = (high + low) / 2.0
    
    # Calculate SMMA (Smoothed Moving Average) - equivalent to RMA/Wilder's smoothing
    def smma(source, period):
        if len(source) < period:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Price) / Period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaws_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply smoothing offsets (jaws +8, teeth +5, lips +3)
    jaws = np.full_like(jaws_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaws_raw) > 8:
        jaws[8:] = jaws_raw[:-8]
    if len(teeth_raw) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips_raw) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaws[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator bullish alignment (jaws < teeth < lips) AND close > EMA34(1d) AND volume spike
            if (jaws[i] < teeth[i] and 
                teeth[i] < lips[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator bearish alignment (jaws > teeth > lips) AND close < EMA34(1d) AND volume spike
            elif (jaws[i] > teeth[i] and 
                  teeth[i] > lips[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks bullish OR close < EMA34(1d) (trend flip)
            if not (jaws[i] < teeth[i] and teeth[i] < lips[i]) or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks bearish OR close > EMA34(1d) (trend flip)
            if not (jaws[i] > teeth[i] and teeth[i] > lips[i]) or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals