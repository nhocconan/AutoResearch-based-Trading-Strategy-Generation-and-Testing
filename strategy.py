#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# In trending markets (price > EMA34), we trade breakouts: long on R3 breakout, short on S3 breakout.
# In ranging markets (price near EMA34), we fade extremes: short near R3, long near S3.
# Volume confirmation (>2.0x 20-period EMA) filters false signals. Designed for 6h timeframe targeting 50-150 total trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "6h_Camarilla_R3S3_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align 1d indicators to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA (stricter to reduce trades)
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Determine market state: trending (price > EMA34) or ranging (price near EMA34)
            price_to_ema_ratio = close[i] / ema_34_aligned[i]
            
            if price_to_ema_ratio > 1.02:  # Strong uptrend (>2% above EMA34)
                # Trending market: trade breakouts in trend direction
                if (close[i] > camarilla_r3_aligned[i] and 
                    volume_confirm):
                    signals[i] = 0.25
                    position = 1
            elif price_to_ema_ratio < 0.98:  # Strong downtrend (>2% below EMA34)
                # Trending market: trade breakouts in trend direction
                if (close[i] < camarilla_s3_aligned[i] and 
                    volume_confirm):
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging market (price within ±2% of EMA34)
                # Ranging market: fade extremes (mean reversion)
                if (close[i] <= camarilla_s3_aligned[i] and 
                    volume_confirm):
                    signals[i] = 0.25
                    position = 1
                elif (close[i] >= camarilla_r3_aligned[i] and 
                      volume_confirm):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price retouches midpoint between R3 and S3 OR EMA34 cross below OR volume drops
            mid = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if (close[i] <= mid or 
                close[i] < ema_34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches midpoint between R3 and S3 OR EMA34 cross above OR volume drops
            mid = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if (close[i] >= mid or 
                close[i] > ema_34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals