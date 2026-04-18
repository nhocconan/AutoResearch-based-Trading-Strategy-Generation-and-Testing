#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly VWAP Reversion with Volume Spike and 1w Trend Filter
# In ranging markets, price tends to revert to the weekly VWAP. Strong volume spikes
# indicate institutional interest and higher probability of mean reversion.
# Weekly EMA34 filter ensures we only take reversions aligned with the longer-term trend,
# avoiding counter-trend trades in strong trends. Designed for low trade frequency.
# Works in both bull and bear markets by fading extremes with institutional volume.

name = "1d_WeeklyVWAP_Reversion_VolumeSpike_TrendFilter"
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
    
    # Get weekly data for VWAP and EMA (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly VWAP (volume-weighted average price)
    # VWAP = sum(price * volume) / sum(volume) for the week
    typical_price = (high + low + close) / 3
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = np.where(cum_vol > 0, cum_pv / cum_vol, np.nan)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike detector: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    # Align weekly VWAP to daily timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap)
    
    # Session filter: 08-20 UTC (trade during active hours)
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Wait for EMA and VWAP warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below VWAP (oversold) + volume spike + weekly uptrend
            if vol_spike[i] and close[i] < vwap_aligned[i] and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price above VWAP (overbought) + volume spike + weekly downtrend
            elif vol_spike[i] and close[i] > vwap_aligned[i] and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to VWAP (mean reversion complete) or trend breaks
            if close[i] >= vwap_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to VWAP or trend breaks
            if close[i] <= vwap_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals