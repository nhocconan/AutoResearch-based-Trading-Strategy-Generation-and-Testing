#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND close > EMA34(1d) AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S3 AND close < EMA34(1d) AND volume > 2.0x 20-period average
# Exit when price retracement to Camarilla pivot point OR EMA34(1d) trend flip
# Uses 12h primary timeframe with 1d HTF for trend filter to reduce whipsaw
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    if len(high) >= 2 and len(low) >= 2 and len(close) >= 2:
        # Use previous bar's OHLC for today's Camarilla levels
        phigh = pd.Series(high).shift(1).values
        plow = pd.Series(low).shift(1).values
        pclose = pd.Series(close).shift(1).values
        
        pivot = (phigh + plow + pclose) / 3.0
        range_val = phigh - plow
        
        # Camarilla levels
        camarilla_h4 = pivot + (range_val * 1.1 / 2)
        camarilla_l4 = pivot - (range_val * 1.1 / 2)
        camarilla_h3 = pivot + (range_val * 1.1 / 4)
        camarilla_l3 = pivot - (range_val * 1.1 / 4)
        camarilla_h2 = pivot + (range_val * 1.1 / 6)
        camarilla_l2 = pivot - (range_val * 1.1 / 6)
        camarilla_h1 = pivot + (range_val * 1.1 / 12)
        camarilla_l1 = pivot - (range_val * 1.1 / 12)
        
        # For breakout: use H3/L3 (R3/S3)
        camarilla_r3 = camarilla_h3
        camarilla_s3 = camarilla_l3
        camarilla_pivot = pivot
    else:
        camarilla_r3 = np.full(n, np.nan)
        camarilla_s3 = np.full(n, np.nan)
        camarilla_pivot = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_pivot[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND close > EMA34(1d) AND volume spike
            if (high[i] > camarilla_r3[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND close < EMA34(1d) AND volume spike
            elif (low[i] < camarilla_s3[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retracement to Camarilla pivot OR close < EMA34(1d) (trend flip)
            if close[i] <= camarilla_pivot[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retracement to Camarilla pivot OR close > EMA34(1d) (trend flip)
            if close[i] >= camarilla_pivot[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals