#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume spike confirmation.
# Long when: Alligator jaws < teeth < lips (bullish alignment) AND price > lips AND 1w close > 1w EMA34 AND 1d volume > 2x 20-period average
# Short when: Alligator jaws > teeth > lips (bearish alignment) AND price < lips AND 1w close < 1w EMA34 AND 1d volume > 2x 20-period average
# Uses discrete sizing 0.25. Target: 7-25 trades/year on 1d.
# Alligator identifies trend emergence, 1w EMA34 filters for higher timeframe trend alignment, volume spike confirms conviction.
# Works in bull (catching trends early) and bear (catching breakdowns early) by trading with the aligned trend.

name = "1d_Alligator_1wTrend_VolumeSpike_v1"
timeframe = "1d"
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
    
    # Load 1d data ONCE before loop for Alligator (SMAs of median price)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Williams Alligator on 1d: SMAs of median price
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    jaws = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values    # 8-period SMA
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values     # 5-period SMA
    
    # Align Alligator lines to 1d primary timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1d volume average (20-period) for volume spike confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
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
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_1d_aligned[i]
        curr_jaws = jaws_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_lips = lips_aligned[i]
        curr_ema_34 = ema_34_aligned[i]
        
        # Volume spike: current 1d volume > 2x 20-period average
        volume_spike = curr_vol > (curr_vol_ma * 2.0)
        
        # Alligator alignment
        bullish_alignment = (curr_jaws < curr_teeth) and (curr_teeth < curr_lips)
        bearish_alignment = (curr_jaws > curr_teeth) and (curr_teeth > curr_lips)
        
        # 1w trend filter: price above/below EMA34
        uptrend_1w = curr_close > curr_ema_34
        downtrend_1w = curr_close < curr_ema_34
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator AND price above lips AND 1w uptrend AND volume spike
            if (bullish_alignment and 
                curr_close > curr_lips and 
                uptrend_1w and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator AND price below lips AND 1w downtrend AND volume spike
            elif (bearish_alignment and 
                  curr_close < curr_lips and 
                  downtrend_1w and 
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