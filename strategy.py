#!/usr/bin/env python3
"""
1h_VolumeSpike_Breakout_4hEMA20_Trend_1dRegime
Hypothesis: For 1h timeframe, use 4h EMA20 for trend direction and 1d Camarilla H3/L3 for structure, with volume spike confirmation on 1h for entry timing. Target 15-35 trades/year by requiring confluence of 4h trend, 1d breakout level, and 1h volume spike. Works in bull markets via breakout continuation and bear markets via mean reversion at extreme levels (H4/L4) when 1d regime is choppy.
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
    
    # 4h data for EMA20 trend filter (loaded ONCE)
    df_4h = get_htf_data(prices, '4h')
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d data for Camarilla calculation (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H3, L3, H4, L4 for breakout and extreme reversal
    camarilla_range = prev_high - prev_low
    h3 = prev_close + camarilla_range * 1.1 / 4
    l3 = prev_close - camarilla_range * 1.1 / 4
    h4 = prev_close + camarilla_range * 1.1 / 2
    l4 = prev_close - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # 1h volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # 1h ATR for stoploss
    tr0 = np.abs(high - low)
    tr1 = np.abs(high[1:] - close[:-1])
    tr2 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr0[0]], np.maximum(tr1, tr2)])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1h chop regime filter: BBW percentile < 30% = choppy (mean revert), > 70% = trending (follow)
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    choppy_market = bb_width_percentile < 30  # choppy = mean revert
    trending_market = bb_width_percentile > 70  # trending = follow breakout
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 4h EMA (20), volume MA (20), ATR (14), BB (100)
    start_idx = max(20, 20, 14, 100)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals
            # In trending market: follow 4h EMA20 direction with 1d H3/L3 breakout
            # In choppy market: mean revert at 1d H4/L4 extremes
            long_breakout = curr_high > h3_aligned[i]
            short_breakout = curr_low < l3_aligned[i]
            long_extreme = curr_low < l4_aligned[i]  # price at extreme low
            short_extreme = curr_high > h4_aligned[i]  # price at extreme high
            
            # Trend filter: price must be on correct side of 4h EMA20
            long_trend = curr_close > ema_20_4h_aligned[i]
            short_trend = curr_close < ema_20_4h_aligned[i]
            
            if trending_market[i]:
                # Trending market: follow breakout with trend
                long_entry = (long_breakout and volume_spike[i] and long_trend)
                short_entry = (short_breakout and volume_spike[i] and short_trend)
            else:
                # Choppy market: mean revert from extremes
                long_entry = (long_extreme and volume_spike[i])
                short_entry = (short_extreme and volume_spike[i])
            
            if long_entry:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # 1. Price closes below 4h EMA20 (trend fail)
            # 2. Price reaches opposite Camarilla level (take profit)
            # 3. ATR stoploss
            atr_stop = entry_price - 2.0 * atr[i]
            if (curr_close < ema_20_4h_aligned[i] or 
                curr_close > l3_aligned[i] or  # took profit at opposite level
                curr_close < atr_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit conditions
            # 1. Price closes above 4h EMA20 (trend fail)
            # 2. Price reaches opposite Camarilla level (take profit)
            # 3. ATR stoploss
            atr_stop = entry_price + 2.0 * atr[i]
            if (curr_close > ema_20_4h_aligned[i] or 
                curr_close < h3_aligned[i] or  # took profit at opposite level
                curr_close > atr_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolumeSpike_Breakout_4hEMA20_Trend_1dRegime"
timeframe = "1h"
leverage = 1.0