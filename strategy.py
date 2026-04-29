#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation
# Long: Alligator bullish (jaw < teeth < lips) AND price > 1d EMA50 AND volume > 1.8x 20-bar avg
# Short: Alligator bearish (jaw > teeth > lips) AND price < 1d EMA50 AND volume > 1.8x 20-bar avg
# Exit: Close crosses Alligator midpoint OR price crosses 1d EMA50 OR ATR stoploss (2.0 * ATR)
# Williams Alligator: jaw=SMA(13,8), teeth=SMA(8,5), lips=SMA(5,3)
# Works in bull via trend continuation, in bear via mean reversion at extremes
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
# Discrete position sizing: 0.25 for long/short, 0.0 for flat to minimize fee churn

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Williams Alligator components
    # Jaw: SMA(13, 8) - median price smoothed with 8-period SMA
    # Teeth: SMA(8, 5) - median price smoothed with 5-period SMA  
    # Lips: SMA(5, 3) - median price smoothed with 3-period SMA
    median_price = (high + low) / 2.0
    
    # Jaw: SMA(13) of median price, then SMA(8) of that
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    
    # Teeth: SMA(8) of median price, then SMA(5) of that
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    # Lips: SMA(5) of median price, then SMA(3) of that
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 14, 21)  # warmup for indicators (21 for Alligator)
    
    for i in range(start_idx, n):
        # Williams Alligator values
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Alligator bullish: jaw < teeth < lips
        alligator_bullish = jaw_val < teeth_val < lips_val
        # Alligator bearish: jaw > teeth > lips
        alligator_bearish = jaw_val > teeth_val > lips_val
        
        # Alligator midpoint (average of jaw and lips)
        alligator_mid = (jaw_val + lips_val) / 2.0
        
        curr_close = close[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.8 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: Close below Alligator mid OR price below 1d EMA50 OR stoploss hit
            if curr_close < alligator_mid or curr_close < curr_ema_1d or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: Close above Alligator mid OR price above 1d EMA50 OR stoploss hit
            if curr_close > alligator_mid or curr_close > curr_ema_1d or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Alligator bullish AND price > 1d EMA50 AND volume spike
            if (alligator_bullish and 
                curr_close > curr_ema_1d and
                vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Alligator bearish AND price < 1d EMA50 AND volume spike
            elif (alligator_bearish and 
                  curr_close < curr_ema_1d and
                  vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals