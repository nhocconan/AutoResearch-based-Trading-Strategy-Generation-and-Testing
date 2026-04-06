#!/usr/bin/env python3
"""
6h Weekly Pivot + Volume Confirmation + ATR Stop v2
Hypothesis: Focus on weekly pivot breakouts (R4/S4) for trend continuation with volume confirmation and 1d trend filter. 
Uses mean reversion at R3/S3 only when price is near weekly pivot AND volume confirms. 
Improved version with stricter entry conditions to increase trade frequency while maintaining quality.
Target: 75-200 trades over 4 years (19-50/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weeklypivot_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # 1d EMA50 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 18) / 20
    
    # Trend bias: above EMA = bullish, below = bearish
    trend_bias_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align to 6h timeframe
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # Calculate weekly pivot from 1d data (using previous week's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly pivot: using Friday's OHLC for the week (approximated as last 5 days)
    weekly_high = np.full(len(close_1d), np.nan)
    weekly_low = np.full(len(close_1d), np.nan)
    weekly_close = np.full(len(close_1d), np.nan)
    
    # Calculate weekly values (simplified: using 5-day rolling)
    for i in range(len(close_1d)):
        if i >= 4:
            weekly_high[i] = np.max(high_1d[i-4:i+1])
            weekly_low[i] = np.min(low_1d[i-4:i+1])
            weekly_close[i] = close_1d[i]
    
    # Weekly pivot levels (using previous week's data)
    wp = np.full(len(close_1d), np.nan)
    wr1 = np.full(len(close_1d), np.nan)
    ws1 = np.full(len(close_1d), np.nan)
    wr2 = np.full(len(close_1d), np.nan)
    ws2 = np.full(len(close_1d), np.nan)
    wr3 = np.full(len(close_1d), np.nan)
    ws3 = np.full(len(close_1d), np.nan)
    wr4 = np.full(len(close_1d), np.nan)
    ws4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        if not (np.isnan(weekly_high[i-1]) or np.isnan(weekly_low[i-1]) or np.isnan(weekly_close[i-1])):
            wh = weekly_high[i-1]
            wl = weekly_low[i-1]
            wc = weekly_close[i-1]
            
            p = (wh + wl + wc) / 3.0
            wp[i] = p
            wr1[i] = 2*p - wl
            ws1[i] = 2*p - wh
            wr2[i] = p + (wh - wl)
            ws2[i] = p - (wh - wl)
            wr3[i] = wh + 2*(p - wl)
            ws3[i] = wl - 2*(wh - p)
            wr4[i] = 3*p - 2*wl
            ws4[i] = 3*wh - 2*wl
    
    # Align weekly pivot levels to 6h timeframe
    wp_aligned = align_htf_to_ltf(prices, df_1d, wp)
    wr1_aligned = align_htf_to_ltf(prices, df_1d, wr1)
    ws1_aligned = align_htf_to_ltf(prices, df_1d, ws1)
    wr2_aligned = align_htf_to_ltf(prices, df_1d, wr2)
    ws2_aligned = align_htf_to_ltf(prices, df_1d, ws2)
    wr3_aligned = align_htf_to_ltf(prices, df_1d, wr3)
    ws3_aligned = align_htf_to_ltf(prices, df_1d, ws3)
    wr4_aligned = align_htf_to_ltf(prices, df_1d, wr4)
    ws4_aligned = align_htf_to_ltf(prices, df_1d, ws4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 30  # Need enough data for weekly calculations
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_bias_aligned[i]) or 
            np.isnan(wr4_aligned[i]) or np.isnan(ws4_aligned[i]) or
            np.isnan(wr3_aligned[i]) or np.isnan(ws3_aligned[i]) or
            np.isnan(wp_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[max(0, i-20):i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below WS3 (mean reversion) OR against 1d trend OR stoploss
            if (close[i] < ws3_aligned[i] or
                trend_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above WR3 (mean reversion) OR against 1d trend OR stoploss
            if (close[i] > wr3_aligned[i] or
                trend_bias_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries with stricter conditions
            # Minimum holding period: only allow new entry after 24 bars flat (1 day)
            if bars_since_entry >= 24:
                # Breakout entries: WR4/WS4 with trend
                bull_breakout = close[i] > wr4_aligned[i]
                bear_breakout = close[i] < ws4_aligned[i]
                
                # Mean reversion entries: WR3/WS3 counter-trend only when near weekly pivot
                near_weekly_pivot = abs(close[i] - wp_aligned[i]) < (wr1_aligned[i] - ws1_aligned[i]) * 0.3
                
                # Long: breakout with trend OR mean reversion at WS3 with volume (only in bear trend near pivot)
                if (bull_breakout and trend_bias_aligned[i] == 1 and volume_filter) or \
                   (close[i] > ws3_aligned[i] and close[i] < wp_aligned[i] and 
                    near_weekly_pivot and volume_filter and trend_bias_aligned[i] == -1):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown with trend OR mean reversion at WR3 with volume (only in bull trend near pivot)
                elif (bear_breakout and trend_bias_aligned[i] == -1 and volume_filter) or \
                     (close[i] < wr3_aligned[i] and close[i] > wp_aligned[i] and 
                      near_weekly_pivot and volume_filter and trend_bias_aligned[i] == 1):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Weekly Pivot + Volume Confirmation + ATR Stop v2
Hypothesis: Focus on weekly pivot breakouts (R4/S4) for trend continuation with volume confirmation and 1d trend filter. 
Uses mean reversion at R3/S3 only when price is near weekly pivot AND volume confirms. 
Improved version with stricter entry conditions to increase trade frequency while maintaining quality.
Target: 75-200 trades over 4 years (19-50/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weeklypivot_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # 1d EMA50 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 18) / 20
    
    # Trend bias: above EMA = bullish, below = bearish
    trend_bias_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align to 6h timeframe
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # Calculate weekly pivot from 1d data (using previous week's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly pivot: using Friday's OHLC for the week (approximated as last 5 days)
    weekly_high = np.full(len(close_1d), np.nan)
    weekly_low = np.full(len(close_1d), np.nan)
    weekly_close = np.full(len(close_1d), np.nan)
    
    # Calculate weekly values (simplified: using 5-day rolling)
    for i in range(len(close_1d)):
        if i >= 4:
            weekly_high[i] = np.max(high_1d[i-4:i+1])
            weekly_low[i] = np.min(low_1d[i-4:i+1])
            weekly_close[i] = close_1d[i]
    
    # Weekly pivot levels (using previous week's data)
    wp = np.full(len(close_1d), np.nan)
    wr1 = np.full(len(close_1d), np.nan)
    ws1 = np.full(len(close_1d), np.nan)
    wr2 = np.full(len(close_1d), np.nan)
    ws2 = np.full(len(close_1d), np.nan)
    wr3 = np.full(len(close_1d), np.nan)
    ws3 = np.full(len(close_1d), np.nan)
    wr4 = np.full(len(close_1d), np.nan)
    ws4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        if not (np.isnan(weekly_high[i-1]) or np.isnan(weekly_low[i-1]) or np.isnan(weekly_close[i-1])):
            wh = weekly_high[i-1]
            wl = weekly_low[i-1]
            wc = weekly_close[i-1]
            
            p = (wh + wl + wc) / 3.0
            wp[i] = p
            wr1[i] = 2*p - wl
            ws1[i] = 2*p - wh
            wr2[i] = p + (wh - wl)
            ws2[i] = p - (wh - wl)
            wr3[i] = wh + 2*(p - wl)
            ws3[i] = wl - 2*(wh - p)
            wr4[i] = 3*p - 2*wl
            ws4[i] = 3*wh - 2*wl
    
    # Align weekly pivot levels to 6h timeframe
    wp_aligned = align_htf_to_ltf(prices, df_1d, wp)
    wr1_aligned = align_htf_to_ltf(prices, df_1d, wr1)
    ws1_aligned = align_htf_to_ltf(prices, df_1d, ws1)
    wr2_aligned = align_htf_to_ltf(prices, df_1d, wr2)
    ws2_aligned = align_htf_to_ltf(prices, df_1d, ws2)
    wr3_aligned = align_htf_to_ltf(prices, df_1d, wr3)
    ws3_aligned = align_htf_to_ltf(prices, df_1d, ws3)
    wr4_aligned = align_htf_to_ltf(prices, df_1d, wr4)
    ws4_aligned = align_htf_to_ltf(prices, df_1d, ws4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 30  # Need enough data for weekly calculations
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_bias_aligned[i]) or 
            np.isnan(wr4_aligned[i]) or np.isnan(ws4_aligned[i]) or
            np.isnan(wr3_aligned[i]) or np.isnan(ws3_aligned[i]) or
            np.isnan(wp_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[max(0, i-20):i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below WS3 (mean reversion) OR against 1d trend OR stoploss
            if (close[i] < ws3_aligned[i] or
                trend_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above WR3 (mean reversion) OR against 1d trend OR stoploss
            if (close[i] > wr3_aligned[i] or
                trend_bias_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries with stricter conditions
            # Minimum holding period: only allow new entry after 24 bars flat (1 day)
            if bars_since_entry >= 24:
                # Breakout entries: WR4/WS4 with trend
                bull_breakout = close[i] > wr4_aligned[i]
                bear_breakout = close[i] < ws4_aligned[i]
                
                # Mean reversion entries: WR3/WS3 counter-trend only when near weekly pivot
                near_weekly_pivot = abs(close[i] - wp_aligned[i]) < (wr1_aligned[i] - ws1_aligned[i]) * 0.3
                
                # Long: breakout with trend OR mean reversion at WS3 with volume (only in bear trend near pivot)
                if (bull_breakout and trend_bias_aligned[i] == 1 and volume_filter) or \
                   (close[i] > ws3_aligned[i] and close[i] < wp_aligned[i] and 
                    near_weekly_pivot and volume_filter and trend_bias_aligned[i] == -1):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown with trend OR mean reversion at WR3 with volume (only in bull trend near pivot)
                elif (bear_breakout and trend_bias_aligned[i] == -1 and volume_filter) or \
                     (close[i] < wr3_aligned[i] and close[i] > wp_aligned[i] and 
                      near_weekly_pivot and volume_filter and trend_bias_aligned[i] == 1):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals