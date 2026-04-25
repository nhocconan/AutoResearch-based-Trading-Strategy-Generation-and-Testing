#!/usr/bin/env python3
"""
6h Camarilla Pivot Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (R3/S3, R4/S4) from 1d timeframe act as significant support/resistance.
Breakouts above R4 or below S4 with 1d EMA34 trend alignment and 6h volume spike capture strong momentum moves.
Works in bull/bear markets by trading breakouts in the direction of the higher timeframe trend.
Target: 50-150 total trades over 4 years (12-37/year) with signal size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4)
    #          S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    # where C = (H+L+Close)/3 (typical price)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    hl_range = df_1d['high'] - df_1d['low']
    
    r4 = typical_price + (hl_range * 1.1 / 2)
    r3 = typical_price + (hl_range * 1.1 / 4)
    s3 = typical_price - (hl_range * 1.1 / 4)
    s4 = typical_price - (hl_range * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (no extra delay needed for pivot points)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # Calculate 20-period volume MA for 6h volume confirmation
    vol_ma_20_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_6h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        r4_val = r4_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        s4_val = s4_aligned[i]
        vol_ma_6h = vol_ma_20_6h[i]
        
        # Volume confirmation: current 6h volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_6h
        
        # Breakout conditions
        # Long breakout: price closes above R4 with uptrend and volume confirmation
        long_breakout = (curr_close > r4_val and 
                        ema_trend > 0 and  # Uptrend filter (price above EMA would be better but we use EMA slope proxy)
                        volume_confirm)
        # Short breakout: price closes below S4 with downtrend and volume confirmation
        short_breakout = (curr_close < s4_val and 
                         ema_trend < 0 and  # Downtrend filter
                         volume_confirm)
        
        # Alternative trend filter: compare current close to EMA
        # Long: price above EMA34, Short: price below EMA34
        long_trend = curr_close > ema_trend
        short_trend = curr_close < ema_trend
        
        if position == 0:
            # Look for entry signals
            # Long: breakout above R4 AND price above EMA34 AND volume confirmation
            long_entry = (curr_close > r4_val and long_trend and volume_confirm)
            # Short: breakout below S4 AND price below EMA34 AND volume confirmation
            short_entry = (curr_close < s4_val and short_trend and volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below R3 (profit taking) OR price falls below EMA34 (trend change)
            if (curr_close < r3_val or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above S3 (profit taking) OR price rises above EMA34 (trend change)
            if (curr_close > s3_val or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S4_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0