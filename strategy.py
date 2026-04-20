#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with weekly trend filter and volume confirmation
# In bull markets: go long when Alligator lines align bullish (jaw < teeth < lips) and price above lips, with weekly uptrend
# In bear markets: go short when Alligator lines align bearish (jaw > teeth > lips) and price below lips, with weekly downtrend
# Volume filter ensures sufficient participation
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA for trend filter (13-period)
    ema_13w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_34w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_55w = pd.Series(close_1w).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Align weekly EMAs to 12h timeframe
    ema_13w_aligned = align_htf_to_ltf(prices, df_1w, ema_13w)
    ema_34w_aligned = align_htf_to_ltf(prices, df_1w, ema_34w)
    ema_55w_aligned = align_htf_to_ltf(prices, df_1w, ema_55w)
    
    # Calculate 12h Williams Alligator (SMMA of median price)
    median_price = (prices['high'].values + prices['low'].values) / 2.0
    
    # Jaw (13-period SMMA, 8-bar shift)
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.concatenate([np.full(8, np.nan), jaw_raw[:-8]]) if len(jaw_raw) > 8 else np.full_like(jaw_raw, np.nan)
    
    # Teeth (8-period SMMA, 5-bar shift)
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.concatenate([np.full(5, np.nan), teeth_raw[:-5]]) if len(teeth_raw) > 5 else np.full_like(teeth_raw, np.nan)
    
    # Lips (5-period SMMA, 3-bar shift)
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.concatenate([np.full(3, np.nan), lips_raw[:-3]]) if len(lips_raw) > 3 else np.full_like(lips_raw, np.nan)
    
    # Calculate 12h ATR for volatility filter
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Volume filter: 12h volume > 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(55, n):  # Start after Alligator warmup
        # Skip if NaN in indicators
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(ema_13w_aligned[i]) or np.isnan(ema_34w_aligned[i]) or np.isnan(ema_55w_aligned[i]) or np.isnan(atr_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter
        vol_filter = volume[i] > volume_ma_20[i]
        
        # Price and Alligator levels
        price = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        weekly_fast = ema_13w_aligned[i]
        weekly_slow = ema_34w_aligned[i]
        weekly_trend = ema_55w_aligned[i]
        
        if position == 0:
            # Bullish Alligator alignment: jaw < teeth < lips
            bullish_align = jaw_val < teeth_val < lips_val
            # Bearish Alligator alignment: jaw > teeth > lips
            bearish_align = jaw_val > teeth_val > lips_val
            
            # Long: bullish alignment, price above lips, weekly uptrend (fast > slow), with volume
            if bullish_align and price > lips_val and weekly_fast > weekly_slow and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: bearish alignment, price below lips, weekly downtrend (fast < slow), with volume
            elif bearish_align and price < lips_val and weekly_fast < weekly_slow and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below teeth or weekly trend turns down
            if price < teeth_val or weekly_fast < weekly_slow:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above teeth or weekly trend turns up
            if price > teeth_val or weekly_fast > weekly_slow:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_WeeklyTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0