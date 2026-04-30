#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze breakout with 1w trend filter and volume confirmation
# Bollinger Band squeeze (low volatility) precedes explosive moves in both bull and bear markets
# 1w EMA50 ensures alignment with strong weekly trend to avoid counter-trend whipsaws
# Volume spike (2.0x 48-period average) confirms institutional participation on 6h timeframe
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull markets via upside breakouts and bear markets via downside breakouts with trend filter.

name = "6h_BBand_Squeeze_1wEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop (MTF Rule #1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Bollinger Bands (20, 2.0) on 6h
    bb_period = 20
    bb_std = 2.0
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_20 + (bb_std * std_20)
    bb_lower = sma_20 - (bb_std * std_20)
    bb_width = (bb_upper - bb_lower) / sma_20  # Normalized bandwidth
    
    # Bollinger Band Squeeze: bandwidth < 20-period average bandwidth
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ma
    
    # Volume confirmation: volume > 2.0x 48-period average (48*6h = 288h = 12 days)
    vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    volume_spike = volume > (2.0 * vol_ma_48)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 50, 48)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(bb_width[i]) or np.isnan(bb_width_ma[i]) or
            np.isnan(vol_ma_48[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_bb_upper = bb_upper[i]
        curr_bb_lower = bb_lower[i]
        curr_bb_squeeze = bb_squeeze[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require Bollinger Band squeeze, volume spike, and weekly trend alignment
            if curr_bb_squeeze and curr_volume_spike:
                # Bullish entry: break above upper band with weekly uptrend
                if curr_close > curr_bb_upper and curr_close > curr_ema_1w:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below lower band with weekly downtrend
                elif curr_close < curr_bb_lower and curr_close < curr_ema_1w:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below middle band (mean reversion) OR squeeze ends (volatility expansion)
            if curr_close < sma_20[i] or not curr_bb_squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above middle band (mean reversion) OR squeeze ends (volatility expansion)
            if curr_close > sma_20[i] or not curr_bb_squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals