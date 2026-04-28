#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above R3 AND close > 1d EMA34 AND volume > 2x 20-bar avg
# Short when price breaks below S3 AND close < 1d EMA34 AND volume > 2x 20-bar avg
# Exit when price retouches the opposite Camarilla level (S3 for long, R3 for short) or volume drops
# Uses discrete position sizing (0.25) to minimize fee churn
# Target: 20-50 trades/year via tight Camarilla breakout conditions + volume confirmation + trend filter
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need volume MA
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels for previous day using daily OHLC
        # We need the previous completed 1d bar's OHLC
        if i < 24:  # Need at least 24 4h bars (1 day) to calculate
            signals[i] = 0.0
            continue
            
        # Get index of 1d bar that completed at or before current 4h bar
        # Since we're on 4h timeframe, 6 bars = 1 day
        idx_1d = i // 6
        if idx_1d < 1:  # Need previous day
            signals[i] = 0.0
            continue
            
        # Get previous day's OHLC (completed 1d bar)
        prev_day_idx = idx_1d - 1
        if prev_day_idx >= len(df_1d):
            signals[i] = 0.0
            continue
            
        prev_high = df_1d['high'].iloc[prev_day_idx]
        prev_low = df_1d['low'].iloc[prev_day_idx]
        prev_close = df_1d['close'].iloc[prev_day_idx]
        
        # Calculate Camarilla levels
        range_ = prev_high - prev_low
        if range_ <= 0:
            signals[i] = 0.0
            continue
            
        camarilla_r3 = prev_close + range_ * 1.1 / 4
        camarilla_s3 = prev_close - range_ * 1.1 / 4
        
        # Volume confirmation: >2x 20-bar average volume
        if i >= 20:
            volume_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > 2.0 * volume_ma_20
        else:
            volume_confirm = False
        
        # Trend filter: price relative to 1d EMA34
        if not np.isnan(ema_34_1d_aligned[i]):
            trend_filter_long = close[i] > ema_34_1d_aligned[i]
            trend_filter_short = close[i] < ema_34_1d_aligned[i]
        else:
            trend_filter_long = False
            trend_filter_short = False
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R3 AND trend filter long AND volume confirmation
            if close[i] > camarilla_r3 and trend_filter_long and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND trend filter short AND volume confirmation
            elif close[i] < camarilla_s3 and trend_filter_short and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches S3 or volume drops
            if close[i] < camarilla_s3 or not volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price retouches R3 or volume drops
            if close[i] > camarilla_r3 or not volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals