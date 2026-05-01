#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + 1w chop regime filter.
# Uses 1w choppiness index to filter ranging vs trending markets: long only when CHOP < 38.2 (trending),
# short only when CHOP < 38.2, and flat when CHOP > 61.8 (ranging). Donchian breakout provides entry
# in direction of 1w trend (price > 1w EMA50 for long, price < 1w EMA50 for short).
# Volume confirmation on 1d ensures breakout validity. Discrete sizing 0.25 balances return/drawdown.
# Target: 12-37 trades/year (50-150 total over 4 years). Works in bull (follow trend) and bear (avoid ranging).

name = "12h_Donchian20_1dVolumeSpike_1wChopRegime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for volume spike and 1w data for chop regime
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d volume spike: volume > 2.0 * 20-period SMA
    vol_sma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike = df_1d['volume'].values > (2.0 * vol_sma_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # 1w choppiness index: CHOP = 100 * log10(SUM(ATR(1),14) / (log10(HH(14)-LL(14)) / log10(14)))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    atr1 = pd.Series(tr).rolling(window=1, min_periods=1).mean().values  # ATR(1) = TR
    sum_atr_14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    range_14 = hh_14 - ll_14
    
    # Avoid division by zero
    chop_raw = np.where(range_14 > 0, 
                        100 * np.log10(sum_atr_14 / 14) / np.log10(range_14), 
                        50.0)  # neutral when range=0
    
    chop_trending = chop_raw < 38.2   # trending regime
    chop_ranging = chop_raw > 61.8    # ranging regime
    
    chop_trending_aligned = align_htf_to_ltf(prices, df_1w, chop_trending)
    chop_ranging_aligned = align_htf_to_ltf(prices, df_1w, chop_ranging)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and 1w indicators
    
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
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_spike_aligned[i]) or
            np.isnan(chop_trending_aligned[i]) or np.isnan(chop_ranging_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol_spike = vol_spike_aligned[i]
        curr_chop_trending = chop_trending_aligned[i]
        curr_chop_ranging = chop_ranging_aligned[i]
        curr_ema_50 = ema_50_1w_aligned[i]
        curr_upper = highest_20[i]
        curr_lower = lowest_20[i]
        
        # Determine 1w trend bias
        bullish_bias = curr_close > curr_ema_50
        bearish_bias = curr_close < curr_ema_50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Only trade in trending regime, avoid ranging
            if curr_chop_trending and not curr_chop_ranging:
                # Long: Donchian breakout above upper band + bullish bias + volume spike
                if (curr_high > curr_upper and 
                    bullish_bias and 
                    curr_vol_spike):
                    signals[i] = 0.25
                    position = 1
                # Short: Donchian breakout below lower band + bearish bias + volume spike
                elif (curr_low < curr_lower and 
                      bearish_bias and 
                      curr_vol_spike):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # In ranging regime or choppy: stay flat
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Donchian breakdown below lower band OR chop turns ranging
            if (curr_low < curr_lower or 
                curr_chop_ranging or 
                not curr_chop_trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Donchian breakout above upper band OR chop turns ranging
            if (curr_high > curr_upper or 
                curr_chop_ranging or 
                not curr_chop_trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals