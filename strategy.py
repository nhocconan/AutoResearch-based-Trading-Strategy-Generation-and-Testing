#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d Camarilla pivot levels with volume confirmation
# Camarilla levels provide statistically significant support/resistance (H4/L4 levels)
# Volume breakout confirms institutional interest
# Timeframe filter (08-20 UTC) avoids low-liquidity periods
# Target: 15-35 trades/year (~60-140 total over 4 years)
# Works in bull/bear: breakouts work in trends, mean reversion at extremes works in ranges

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels (using previous day's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # H4 = C + (H-L) * 1.1/2
    # L4 = C - (H-L) * 1.1/2
    # H3 = C + (H-L) * 1.1/4
    # L3 = C - (H-L) * 1.1/4
    # H2 = C + (H-L) * 1.1/6
    # L2 = C - (H-L) * 1.1/6
    # H1 = C + (H-L) * 1.1/12
    # L1 = C - (H-L) * 1.1/12
    
    # For each bar, use previous day's OHLC to calculate levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h4 = np.zeros(len(close_1d))
    camarilla_l4 = np.zeros(len(close_1d))
    camarilla_h3 = np.zeros(len(close_1d))
    camarilla_l3 = np.zeros(len(close_1d))
    camarilla_h2 = np.zeros(len(close_1d))
    camarilla_l2 = np.zeros(len(close_1d))
    camarilla_h1 = np.zeros(len(close_1d))
    camarilla_l1 = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        h = high_1d[i]
        l = low_1d[i]
        c = close_1d[i]
        rang = h - l
        if rang <= 0:
            camarilla_h4[i] = camarilla_l4[i] = camarilla_h3[i] = camarilla_l3[i] = camarilla_h2[i] = camarilla_l2[i] = camarilla_h1[i] = camarilla_l1[i] = c
        else:
            camarilla_h4[i] = c + rang * 1.1 / 2
            camarilla_l4[i] = c - rang * 1.1 / 2
            camarilla_h3[i] = c + rang * 1.1 / 4
            camarilla_l3[i] = c - rang * 1.1 / 4
            camarilla_h2[i] = c + rang * 1.1 / 6
            camarilla_l2[i] = c - rang * 1.1 / 6
            camarilla_h1[i] = c + rang * 1.1 / 12
            camarilla_l1[i] = c - rang * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (using previous day's levels)
    h4_1h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_1h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_1h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_1h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_1h = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_1h = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_1h = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_1h = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Get 4h data for trend filter (optional but helps reduce whipsaws)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        # Fallback to simple EMA on 1h if 4h data insufficient
        close_series = pd.Series(close)
        ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
        use_4h_filter = False
    else:
        close_4h = df_4h['close'].values
        close_4h_series = pd.Series(close_4h)
        ema_20_4h = close_4h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
        ema_20_1h = align_htf_to_ltf(prices, df_4h, ema_20_4h)
        use_4h_filter = True
    
    # Volume filter: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma_20 * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% of capital
    
    for i in range(20, n):
        # Skip if not in trading session
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Skip if volume filter not met
        if not volume_filter[i]:
            signals[i] = 0.0
            continue
        
        # Skip if Camarilla levels not ready
        if (np.isnan(h4_1h[i]) or np.isnan(l4_1h[i]) or 
            np.isnan(h3_1h[i]) or np.isnan(l3_1h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter (if 4h data available)
        if use_4h_filter:
            if np.isnan(ema_20_1h[i]):
                signals[i] = 0.0
                continue
            above_ema = close[i] > ema_20_1h[i]
            below_ema = close[i] < ema_20_1h[i]
        else:
            # Use price action for trend: above/below recent swing
            lookback = min(10, i)
            if lookback < 2:
                signals[i] = 0.0
                continue
            recent_high = np.max(high[i-lookback:i+1])
            recent_low = np.min(low[i-lookback:i+1])
            above_ema = close[i] > (recent_high + recent_low) / 2
            below_ema = close[i] < (recent_high + recent_low) / 2
        
        # Camarilla breakout conditions
        # Long when price breaks above H3/H4 with volume
        long_breakout = close[i] > h3_1h[i]
        # Short when price breaks below L3/L4 with volume
        short_breakout = close[i] < l3_1h[i]
        
        # Entry conditions: breakout in direction of trend
        long_entry = long_breakout and above_ema
        short_entry = short_breakout and below_ema
        
        # Exit conditions: opposite breakout or mean reversion to mid levels
        exit_long = position == 1 and (close[i] < l3_1h[i] or close[i] < (h3_1h[i] + l3_1h[i]) / 2)
        exit_short = position == -1 and (close[i] > h3_1h[i] or close[i] > (h3_1h[i] + l3_1h[i]) / 2)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_camarilla_volume_filter"
timeframe = "1h"
leverage = 1.0