#!/usr/bin/env python3
"""
1h_Camarilla_H3L3_Breakout_4hTrend_VolumeConfirm
Hypothesis: On 1h timeframe, use Camarilla pivot (H3/L3) breakout filtered by 4h EMA34 trend and volume spikes (>2.0x 24-bar average).
Camarilla provides intraday support/resistance levels; 4h EMA34 ensures alignment with higher timeframe trend; volume confirms breakout strength.
Designed for low trade frequency (target: 60-150 total trades over 4 years) to minimize fee drag. Works in bull markets via breakouts and in bear markets via failed breaks near H3/L3.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (precomputed)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for EMA34 trend filter (loaded ONCE)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA34 trend filter (loaded ONCE)
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Daily data for Camarilla pivot (H3, L3, H4, L4) - using 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # H4 = Close + 1.1 * (High - Low) * 1.1/2
    # L4 = Close - 1.1 * (High - Low) * 1.1/2
    # H3 = Close + 1.1 * (High - Low) * 1.1/4
    # L3 = Close - 1.1 * (High - Low) * 1.1/4
    # H2 = Close + 1.1 * (High - Low) * 1.1/6
    # L2 = Close - 1.1 * (High - Low) * 1.1/6
    # H1 = Close + 1.1 * (High - Low) * 1.1/12
    # L1 = Close - 1.1 * (High - Low) * 1.1/12
    
    prev_close = df_1d['close'].shift(1).values  # Previous day close
    prev_high = df_1d['high'].shift(1).values    # Previous day high
    prev_low = df_1d['low'].shift(1).values      # Previous day low
    
    # Calculate Camarilla levels
    H4 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 2
    L4 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 2
    H3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
    L3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    H2 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 6
    L2 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 6
    H1 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 12
    L1 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume spike: current volume > 2.0 * 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34 (34), volume MA (24), and Camarilla (need previous day)
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 + volume spike + 4h uptrend
            long_breakout = curr_close > H3_aligned[i]
            # Short: price breaks below L3 + volume spike + 4h downtrend
            short_breakout = curr_close < L3_aligned[i]
            
            # Trend filter: price must be on correct side of 4h EMA34
            long_trend = curr_close > ema_34_4h_aligned[i]
            short_trend = curr_close < ema_34_4h_aligned[i]
            
            long_entry = long_breakout and volume_spike[i] and long_trend
            short_entry = short_breakout and volume_spike[i] and short_trend
            
            if long_entry:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below L3 OR trend reverses
            if curr_close < L3_aligned[i] or curr_close < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit when price closes above H3 OR trend reverses
            if curr_close > H3_aligned[i] or curr_close > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hTrend_VolumeConfirm"
timeframe = "1h"
leverage = 1.0