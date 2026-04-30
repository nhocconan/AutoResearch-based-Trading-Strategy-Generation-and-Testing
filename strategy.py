#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout with 12h EMA50 Trend Filter and Volume Confirmation
# Camarilla pivot levels (R3/S3) act as strong support/resistance - breakouts indicate institutional interest
# 12h EMA50 provides medium-term trend filter to avoid counter-trend trades
# Volume confirmation (>2.0x average) ensures breakout legitimacy
# Works in bull/bear: breakouts occur in all regimes, volume confirms legitimacy, trend filter reduces false signals
# Discrete sizing (0.25) targets 75-200 total trades over 4 years (19-50/year) to avoid fee drag

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels (R3, S3) from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Range = high - low
    price_range = high - low
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3 = close + (price_range * 1.1 / 4.0)
    camarilla_s3 = close - (price_range * 1.1 / 4.0)
    
    # Need previous day's levels - shift by 1
    camarilla_r3_prev = np.roll(camarilla_r3, 1)
    camarilla_s3_prev = np.roll(camarilla_s3, 1)
    camarilla_r3_prev[0] = np.nan
    camarilla_s3_prev[0] = np.nan
    
    # Breakout conditions
    breakout_up = close > camarilla_r3_prev
    breakout_down = close < camarilla_s3_prev
    
    # Volume confirmation: volume > 2.0x 24-period average (1 day on 4h chart)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_ma_24)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 24)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3_prev[i]) or 
            np.isnan(camarilla_s3_prev[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price above R3 + above 12h EMA50
                if curr_breakout_up and curr_close > curr_ema_50_12h:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below S3 + below 12h EMA50
                elif curr_breakout_down and curr_close < curr_ema_50_12h:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price closes below S3 (reversal) or breaks above R3 (take profit)
            if curr_close < camarilla_s3_prev[i] or curr_close > camarilla_r3_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above R3 (reversal) or breaks below S3 (take profit)
            if curr_close > camarilla_r3_prev[i] or curr_close < camarilla_s3_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals