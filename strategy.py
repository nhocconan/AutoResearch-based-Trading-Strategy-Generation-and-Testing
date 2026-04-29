#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 1h timeframe with 4h HTF for signal direction (reduces noise, improves Sharpe)
# 4h EMA50 provides trend filter for regime alignment (works in bull/bear)
# Volume > 2.0x average confirms institutional participation
# Discrete position sizing (0.20) with Camarilla H4/L4 mean reversion exit
# Session filter (08-20 UTC) reduces off-hours noise
# Target: ~20-40 trades/year to stay within fee drag limits

name = "1h_Camarilla_H3L3_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla pivot levels (based on previous bar's OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla calculations based on previous bar
    # Pivot point = (H + L + C) / 3
    pp = (high_4h + low_4h + close_4h) / 3.0
    # H3 = C + (H - L) * 1.1 / 4
    h3 = close_4h + (high_4h - low_4h) * 1.1 / 4.0
    # L3 = C - (H - L) * 1.1 / 4
    l3 = close_4h - (high_4h - low_4h) * 1.1 / 4.0
    # H4 = C + (H - L) * 1.1 / 2
    h4 = close_4h + (high_4h - low_4h) * 1.1 / 2.0
    # L4 = C - (H - L) * 1.1 / 2
    l4 = close_4h - (high_4h - low_4h) * 1.1 / 2.0
    
    # Use previous bar's values (shift by 1) to avoid look-ahead
    pp_shifted = np.roll(pp, 1)
    h3_shifted = np.roll(h3, 1)
    l3_shifted = np.roll(l3, 1)
    h4_shifted = np.roll(h4, 1)
    l4_shifted = np.roll(l4, 1)
    pp_shifted[0] = np.nan
    h3_shifted[0] = np.nan
    l3_shifted[0] = np.nan
    h4_shifted[0] = np.nan
    l4_shifted[0] = np.nan
    
    # Align 4h indicators to 1h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_4h, pp_shifted)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3_shifted)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3_shifted)
    h4_aligned = align_htf_to_ltf(prices, df_4h, h4_shifted)
    l4_aligned = align_htf_to_ltf(prices, df_4h, l4_shifted)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Volume and 4h EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_pp = pp_aligned[i]
        curr_h3 = h3_aligned[i]
        curr_l3 = l3_aligned[i]
        curr_h4 = h4_aligned[i]
        curr_l4 = l4_aligned[i]
        curr_ema50_4h = ema_50_4h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below H4 (mean reversion to H4 level)
            if curr_close < curr_h4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price above L4 (mean reversion to L4 level)
            if curr_close > curr_l4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above H3, 4h EMA50 up-trend, volume confirmed
            if curr_high > curr_h3 and curr_close > curr_ema50_4h and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below L3, 4h EMA50 down-trend, volume confirmed
            elif curr_low < curr_l3 and curr_close < curr_ema50_4h and vol_confirmed:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals