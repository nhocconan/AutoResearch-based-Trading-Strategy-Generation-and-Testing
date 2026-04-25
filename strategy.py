#!/usr/bin/env python3
"""
4h Camarilla R3S3 Breakout + 1d EMA34 Trend Filter + Volume Spike
Hypothesis: Camarilla pivot levels identify intraday support/resistance. 
Breakout of R3/S3 levels with 1d EMA34 trend alignment and volume confirmation 
captures strong momentum moves. Works in bull/bear via trend filter. 
Target: 20-50 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (high + low + close) / 3
    # Range = H - L
    price_range = high - low
    
    # Camarilla R3, S3 levels (based on previous bar's typical price and range)
    # R3 = typical_price + (price_range * 1.1/2)
    # S3 = typical_price - (price_range * 1.1/2)
    # But we need previous day's values, so shift by 1
    prev_typical = pd.Series(typical_price).shift(1).values
    prev_range = pd.Series(price_range).shift(1).values
    
    camarilla_r3 = prev_typical + (prev_range * 1.1 / 2)
    camarilla_s3 = prev_typical - (prev_range * 1.1 / 2)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and Camarilla calculation
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Camarilla breakout conditions
        breakout_long = curr_high > camarilla_r3[i]  # Break above R3
        breakout_short = curr_low < camarilla_s3[i]  # Break below S3
        
        # Trend filter: price above/below 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + trend alignment + volume
            long_entry = breakout_long and uptrend and vol_spike
            short_entry = breakout_short and downtrend and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price retouches Camarilla H3 level OR trend reverses
            camarilla_h3 = prev_typical[i] + (prev_range[i] * 1.1/4) if not (np.isnan(prev_typical[i]) or np.isnan(prev_range[i])) else camarilla_r3[i] * 0.8
            if curr_close < camarilla_h3 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price retouches Camarilla L3 level OR trend reverses
            camarilla_l3 = prev_typical[i] - (prev_range[i] * 1.1/4) if not (np.isnan(prev_typical[i]) or np.isnan(prev_range[i])) else camarilla_s3[i] * 1.2
            if curr_close > camarilla_l3 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0