#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume spike confirmation.
# Long when: Alligator jaws < teeth < lips (bullish alignment) AND price > lips AND 1d close > 1d EMA34 AND 6h volume > 2x 20-period average
# Short when: Alligator jaws > teeth > lips (bearish alignment) AND price < lips AND 1d close < 1d EMA34 AND 6h volume > 2x 20-period average
# Uses discrete sizing 0.25. Target: 12-37 trades/year on 6h.
# Alligator identifies trend emergence, 1d EMA34 filters for higher timeframe trend alignment, volume spike confirms conviction.
# Works in bull (catching trends early) and bear (catching breakdowns early) by trading with the aligned trend.

name = "6h_Alligator_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 6h data ONCE before loop for Alligator (SMAs of median price)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator on 6h: SMAs of median price
    median_price_6h = (df_6h['high'].values + df_6h['low'].values) / 2
    jaws = pd.Series(median_price_6h).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(median_price_6h).rolling(window=8, min_periods=8).mean().values    # 8-period SMA
    lips = pd.Series(median_price_6h).rolling(window=5, min_periods=5).mean().values     # 5-period SMA
    
    # Align Alligator lines to 6h primary timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_6h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 6h volume average (20-period) for volume spike confirmation
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA and Alligator
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_6h_aligned[i]
        curr_jaws = jaws_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_lips = lips_aligned[i]
        curr_ema_34 = ema_34_aligned[i]
        
        # Volume spike: current 6h volume > 2x 20-period average
        volume_spike = curr_vol > (curr_vol_ma * 2.0)
        
        # Alligator alignment
        bullish_alignment = (curr_jaws < curr_teeth) and (curr_teeth < curr_lips)
        bearish_alignment = (curr_jaws > curr_teeth) and (curr_teeth > curr_lips)
        
        # 1d trend filter: price above/below EMA34
        uptrend_1d = curr_close > curr_ema_34
        downtrend_1d = curr_close < curr_ema_34
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator AND price above lips AND 1d uptrend AND volume spike
            if (bullish_alignment and 
                curr_close > curr_lips and 
                uptrend_1d and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator AND price below lips AND 1d downtrend AND volume spike
            elif (bearish_alignment and 
                  curr_close < curr_lips and 
                  downtrend_1d and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator loses bullish alignment OR price closes below lips
            if (not bullish_alignment or 
                curr_close < curr_lips):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator loses bearish alignment OR price closes above lips
            if (not bearish_alignment or 
                curr_close > curr_lips):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals