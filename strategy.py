#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and 1w EMA34 trend filter
# Uses 12h primary timeframe to target 12-37 trades per year (50-150 total over 4 years).
# Camarilla pivots from 1d provide strong support/resistance. Breakouts beyond R3/S3
# indicate momentum moves. Volume spike (2.0x 20-period average) confirms validity.
# 1w EMA > price for longs, EMA < price for shorts ensures alignment with weekly trend.
# Discrete sizing 0.25 balances risk and minimizes fee churn. Works in bull via breakout longs
# with weekly uptrend, and in bear via breakout shorts with weekly downtrend.

name = "12h_Camarilla_R3S3_Breakout_1dVolumeSpike_1wEMA34_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    # Use previous day's OHLC for today's Camarilla levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r4 = pivot + (prev_high - prev_low) * 1.1 / 2.0
    r3 = pivot + (prev_high - prev_low) * 1.1 / 4.0
    s3 = pivot - (prev_high - prev_low) * 1.1 / 4.0
    s4 = pivot - (prev_high - prev_low) * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 12h timeframe (wait for completed 1d bar)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34)  # warmup for volume MA and weekly EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema_1w = ema_34_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish breakout: price breaks above R3 AND above weekly EMA (uptrend)
                if curr_close > curr_r3 and curr_close > curr_ema_1w:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below S3 AND below weekly EMA (downtrend)
                elif curr_close < curr_s3 and curr_close < curr_ema_1w:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below S3 (mean reversion to pivot area)
            if curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above R3 (mean reversion to pivot area)
            if curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals