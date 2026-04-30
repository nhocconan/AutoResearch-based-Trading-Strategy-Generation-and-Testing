#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation
# Camarilla R1/S1 levels provide intraday support/resistance with lower false breakout rate than tighter levels
# 4h EMA > 50-period ensures alignment with medium-term trend to avoid counter-trend whipsaws
# Volume spike (2.0x 24-period average) confirms institutional participation on 1h timeframe
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Discrete sizing 0.20 minimizes fee churn. Target: 60-150 total trades over 4 years (15-37/year).
# Works in bull markets via breakouts above R1 and bear markets via breakdowns below S1 with trend filter.

name = "1h_Camarilla_R1S1_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
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
    
    # Load 4h data ONCE before loop (MTF Rule #1)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla pivot levels (R1, S1)
    camarilla_r1 = close + ((high - low) * 1.125 / 12)
    camarilla_s1 = close - ((high - low) * 1.125 / 12)
    
    # Volume confirmation: volume > 2.0x 24-period average (24*1h = 1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 24)  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50_4h_aligned[i]
        curr_r1 = camarilla_r1[i]
        curr_s1 = camarilla_s1[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and price above/below EMA for trend alignment
            if curr_volume_spike:
                # Bullish entry: price above EMA and break above R1
                if curr_close > curr_ema and curr_close > curr_r1:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price below EMA and break below S1
                elif curr_close < curr_ema and curr_close < curr_s1:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below EMA (trend change) OR below S1 (mean reversion)
            if curr_close < curr_ema or curr_close < curr_s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price rises above EMA (trend change) OR above R1 (mean reversion)
            if curr_close > curr_ema or curr_close > curr_r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals