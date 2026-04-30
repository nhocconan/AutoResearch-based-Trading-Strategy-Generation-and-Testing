#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# Bollinger Band width percentile detects low volatility squeezes (range-bound conditions).
# Breakouts from squeezes with volume spike and 1d EMA50 trend alignment capture explosive moves.
# Works in bull via breakout longs above upper band, in bear via breakout shorts below lower band.
# Discrete sizing 0.25 balances risk and minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_BB_Squeeze_Breakout_1dEMA50_VolumeSpike_v1"
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
    
    # Calculate 6h Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    close_s = pd.Series(close)
    bb_ma = close_s.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_s.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_ma + (bb_std_dev * bb_std)
    bb_lower = bb_ma - (bb_std_dev * bb_std)
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (50-period lookback) to detect squeeze
    bb_width_s = pd.Series(bb_width)
    bb_width_percentile = bb_width_s.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    squeeze_condition = bb_width_percentile < 20  # Bottom 20% = squeeze
    
    # Calculate 1d EMA(50) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50)  # warmup for BB and 1d EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(squeeze_condition[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bb_upper = bb_upper[i]
        curr_bb_lower = bb_lower[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_squeeze = squeeze_condition[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volatility squeeze breakout with volume confirmation
            if curr_squeeze and curr_volume_spike:
                # Bullish breakout: price breaks above upper band AND above 1d EMA50 (bullish bias)
                if (curr_close > curr_bb_upper and 
                    curr_close > curr_ema_50_1d):
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower band AND below 1d EMA50 (bearish bias)
                elif (curr_close < curr_bb_lower and 
                      curr_close < curr_ema_50_1d):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price drops below middle band (mean reversion) or breaks below lower band (stop)
            if curr_close < bb_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above middle band (mean reversion) or breaks above upper band (stop)
            if curr_close > bb_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals