#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H4/L4 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above 12h Camarilla H4 level AND 1d close > 1d EMA34 (uptrend) AND 12h volume > 1.5x 20-period volume MA.
# Short when price breaks below 12h Camarilla L4 level AND 1d close < 1d EMA34 (downtrend) AND 12h volume > 1.5x 20-period volume MA.
# Exit on retracement to 12h Camarilla H3/L3 levels or trend reversal.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Camarilla levels provide mathematically derived support/resistance, 1d EMA34 filters for higher-timeframe trend alignment, volume confirms participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1d trend when volume confirms.

name = "12h_Camarilla_H4L4_Breakout_1dEMA34_VolumeSpike_Session"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d OHLC for Camarilla levels (use previous day's data to avoid look-ahead)
    df_1d_prev = df_1d.copy()
    df_1d_prev['open'] = df_1d_prev['open'].shift(1)
    df_1d_prev['high'] = df_1d_prev['high'].shift(1)
    df_1d_prev['low'] = df_1d_prev['low'].shift(1)
    df_1d_prev['close'] = df_1d_prev['close'].shift(1)
    
    # Calculate Camarilla levels using previous day's OHLC
    prev_high = df_1d_prev['high'].values
    prev_low = df_1d_prev['low'].values
    prev_close = df_1d_prev['close'].values
    
    # Calculate the range
    range_hl = prev_high - prev_low
    
    # Calculate Camarilla levels
    camarilla_h3 = prev_close + 1.125 * range_hl  # H3 level for exit
    camarilla_l3 = prev_close - 1.125 * range_hl  # L3 level for exit
    camarilla_h4 = prev_close + 1.35 * range_hl   # H4 level for entry
    camarilla_l4 = prev_close - 1.35 * range_hl   # L4 level for entry
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d_prev, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d_prev, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d_prev, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d_prev, camarilla_l4)
    
    # Calculate 12h volume 20-period MA for spike detection
    volume_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(volume_ma_12h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 12h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_12h[i] * 1.5)
        
        # Camarilla breakout conditions
        breakout_up = high_val > camarilla_h4_aligned[i]  # Price breaks above H4 level
        breakout_down = low_val < camarilla_l4_aligned[i]  # Price breaks below L4 level
        
        # 1d trend conditions
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Camarilla breakout up AND 1d uptrend AND volume spike AND session
            if breakout_up and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down AND 1d downtrend AND volume spike AND session
            elif breakout_down and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Camarilla H3/L3 levels OR trend changes
            if close_val < camarilla_h3_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Camarilla H3/L3 levels OR trend changes
            if close_val > camarilla_l3_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals