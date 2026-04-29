#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses Camarilla pivot levels (R3/S3) for high-probability breakout entries with strong institutional interest
# 1d EMA34 provides robust trend filter to avoid counter-trend trades and align with daily momentum
# Volume spike (2.0x 20-period average) confirms breakout validity with strong participation
# Fixed 0.25 position size to minimize fee churn and control drawdown
# Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag
# Works in bull markets via R3 breaks and in bear markets via S3 breaks
# Camarilla levels derived from prior 1d range, making them adaptive to volatility

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels from prior 1d bar (yesterday's H/L/C)
        # Need to get the prior completed 1d bar's data
        if i >= 1:  # we need at least one prior bar to get 1d data
            # Get index of the prior 1d bar in df_1d
            # Since we're on 12h timeframe, each 1d bar = 2 of our bars
            # We want the 1d bar that ended before current bar
            # Convert current index to approximate 1d index
            approx_1d_idx = i // 2  # 2x 12h bars per 1d
            if approx_1d_idx >= 1 and approx_1d_idx < len(df_1d):
                # Use the prior 1d bar (yesterday)
                prior_1d_idx = approx_1d_idx - 1
                if prior_1d_idx >= 0:
                    prior_high = df_1d['high'].iloc[prior_1d_idx]
                    prior_low = df_1d['low'].iloc[prior_1d_idx]
                    prior_close = df_1d['close'].iloc[prior_1d_idx]
                    
                    # Calculate Camarilla levels
                    range_val = prior_high - prior_low
                    camarilla_r3 = prior_close + range_val * 1.1 / 4
                    camarilla_s3 = prior_close - range_val * 1.1 / 4
                else:
                    camarilla_r3 = close[i]
                    camarilla_s3 = close[i]
            else:
                camarilla_r3 = close[i]
                camarilla_s3 = close[i]
        else:
            camarilla_r3 = close[i]
            camarilla_s3 = close[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below 1d EMA34 (trend change) OR price < Camarilla S3 (failed breakout)
            if curr_close < curr_ema_1d or curr_close < camarilla_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above 1d EMA34 (trend change) OR price > Camarilla R3 (failed breakout)
            if curr_close > curr_ema_1d or curr_close > camarilla_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND price > 1d EMA34 AND volume spike
            if curr_high > camarilla_r3 and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S3 AND price < 1d EMA34 AND volume spike
            elif curr_low < camarilla_s3 and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals