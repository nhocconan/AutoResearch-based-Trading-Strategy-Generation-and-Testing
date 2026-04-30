#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 Breakout with 4h EMA200 Trend Filter and Volume Confirmation
# Uses 4h EMA200 for trend direction (avoid counter-trend trades) and 1h for precise entry timing
# Camarilla R3/S3 breakouts with >2.0x volume confirmation capture institutional interest
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Discrete sizing (0.20) targets 60-150 total trades over 4 years (15-37/year) to avoid fee drag
# Works in bull/bear: trend filter adapts to market direction, volume confirms legitimacy

name = "1h_Camarilla_R3S3_Breakout_4hEMA200_Trend_Volume_Session_v1"
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
    
    # Calculate Camarilla pivot levels (R3, S3) from previous 1h bar
    typical_price = (high + low + close) / 3.0
    price_range = high - low
    camarilla_r3 = close + (price_range * 1.1 / 4.0)
    camarilla_s3 = close - (price_range * 1.1 / 4.0)
    
    # Previous bar's levels
    camarilla_r3_prev = np.roll(camarilla_r3, 1)
    camarilla_s3_prev = np.roll(camarilla_s3, 1)
    camarilla_r3_prev[0] = np.nan
    camarilla_s3_prev[0] = np.nan
    
    # Breakout conditions
    breakout_up = close > camarilla_r3_prev
    breakout_down = close < camarilla_s3_prev
    
    # Volume confirmation: volume > 2.0x 24-period average (~1 day on 1h chart)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_ma_24)
    
    # Calculate 4h EMA200 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    ema_200_4h = pd.Series(df_4h['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 24, 200)  # warmup for Camarilla, volume MA, and 4h EMA200
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3_prev[i]) or 
            np.isnan(camarilla_s3_prev[i]) or
            np.isnan(ema_200_4h_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_200_4h = ema_200_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price above R3 + above 4h EMA200 (uptrend)
                if curr_breakout_up and curr_close > curr_ema_200_4h:
                    signals[i] = 0.20
                    position = 1
                # Bearish breakout: price below S3 + below 4h EMA200 (downtrend)
                elif curr_breakout_down and curr_close < curr_ema_200_4h:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price closes below S3 (reversal) or breaks above R3 (take profit)
            if curr_close < camarilla_s3_prev[i] or curr_close > camarilla_r3_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price closes above R3 (reversal) or breaks below S3 (take profit)
            if curr_close > camarilla_r3_prev[i] or curr_close < camarilla_s3_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals