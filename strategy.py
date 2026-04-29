#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Volume-Weighted Average Price (VWAP) Deviation with 1d EMA34 trend filter and volume spike confirmation
# VWAP deviation identifies mean-reversion opportunities: long when price significantly below VWAP in uptrend,
# short when price significantly above VWAP in downtrend. Uses 1d EMA34 for trend bias to avoid counter-trend trades.
# Volume spike (>2x 20-period average) ensures institutional participation. Designed for ~20-30 trades/year on 12h.
# Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Williams %R not used to avoid overtrading.

name = "12h_VWAP_Deviation_1dEMA34_VolumeSpike_v1"
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
    
    # Get 1d data for EMA34 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate VWAP and standard deviation (20-period) on 12h data
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = np.where(cum_vol != 0, cum_pv / cum_vol, typical_price)
    
    # VWAP deviation: (price - VWAP) / VWAP
    vwap_dev = (close - vwap) / vwap
    
    # Calculate 20-period average volume for spike confirmation (on 12h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # EMA34 and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vwap_dev[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vwap_dev = vwap_dev[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits: return to flat when VWAP deviation reverts toward zero
        if position == 1:  # Long position
            if curr_vwap_dev > -0.005:  # Exit when deviation < -0.5% (reverting to VWAP)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if curr_vwap_dev < 0.005:  # Exit when deviation > +0.5% (reverting to VWAP)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume spike confirmation: current volume > 2.0x 20-period average
            vol_spike = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: price significantly below VWAP (-1.5%) in uptrend (price > 1d EMA34)
            if vol_spike and curr_close > curr_ema34_1d:
                if curr_vwap_dev < -0.015:  # -1.5% deviation from VWAP
                    signals[i] = 0.25
                    position = 1
            # Short entry: price significantly above VWAP (+1.5%) in downtrend (price < 1d EMA34)
            elif vol_spike and curr_close < curr_ema34_1d:
                if curr_vwap_dev > 0.015:  # +1.5% deviation from VWAP
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
    
    return signals