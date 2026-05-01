#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Uses Camarilla pivot levels from 1d data: R3/S3 as fade levels, R4/S4 as breakout levels
# Trend filter: price > 1d EMA34 for long bias, price < 1d EMA34 for short bias
# Volume confirmation: volume > 2.0x 20-period EMA to ensure institutional participation
# Designed for low trade frequency: ~15-25 trades/year per symbol with 0.25 sizing
# Camarilla levels work in both bull and bear markets as dynamic support/resistance
# Breakouts in direction of 1d trend have higher follow-through probability

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_v1"
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
    
    # 1d HTF data for Camarilla pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    # Using prior completed day's values to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's values (shifted by 1 to use completed day only)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first value to NaN since we don't have prior day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    camarilla_h5 = prev_close + (range_ * 1.1 / 2)  # R4
    camarilla_h4 = prev_close + (range_ * 1.1 / 4)  # R3
    camarilla_h3 = prev_close + (range_ * 1.1 / 6)  # R2
    camarilla_l3 = prev_close - (range_ * 1.1 / 6)  # S2
    camarilla_l2 = prev_close - (range_ * 1.1 / 4)  # S3
    camarilla_l1 = prev_close - (range_ * 1.1 / 2)  # S4
    
    # Align Camarilla levels to 6h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)  # R3
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)  # S3
    h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)  # R4
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)  # S4
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1d data for EMA34 (34 days) + Camarilla (2 days) + volume EMA20
    start_idx = max(34, 2, 20)  # 34 days for EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(h4_aligned[i]) or np.isnan(l2_aligned[i]) or 
            np.isnan(h5_aligned[i]) or np.isnan(l1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        bullish_bias = close[i] > ema_34_aligned[i]
        bearish_bias = close[i] < ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: break above R4 (h5) with volume spike
                if close[i] > h5_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: break below S4 (l1) with volume spike
                if close[i] < l1_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No clear trend
        
        elif position == 1:  # Long position
            # Exit: close below R3 (h4) - failure of breakout
            if close[i] < h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: close above S3 (l2) - failure of breakdown
            if close[i] > l2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals