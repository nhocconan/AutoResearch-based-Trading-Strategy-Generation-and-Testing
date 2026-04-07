#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot + 12h EMA Trend + Volume Confirmation
# Hypothesis: Camarilla pivot levels on 6h act as strong support/resistance in ranging markets.
# Fade at R3/S3 levels when 12h EMA confirms range-bound market; breakout continuation at R4/S4.
# Works in both bull and bear by following 12h trend direction for breakouts and fading extremes in ranges.
# Target: 20-40 trades/year (80-160 total over 4 years).

name = "6h_camarilla_pivot_12h_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(30) for trend filter
    ema_30_12h = pd.Series(close_12h).ewm(span=30, adjust=False).mean().values
    ema_30_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_30_12h)
    
    # Calculate Camarilla levels for 6h using previous day's OHLC
    # We need daily OHLC to calculate Camarilla, so get 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC (shifted by 1 to avoid look-ahead)
    prev_day_open = df_1d['open'].shift(1).values
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    # Align daily data to 6h
    prev_day_open_6h = align_htf_to_ltf(prices, df_1d, prev_day_open)
    prev_day_high_6h = align_htf_to_ltf(prices, df_1d, prev_day_high)
    prev_day_low_6h = align_htf_to_ltf(prices, df_1d, prev_day_low)
    prev_day_close_6h = align_htf_to_ltf(prices, df_1d, prev_day_close)
    
    # Camarilla levels calculation
    # Range = previous day high - low
    # Close = previous day close
    # Levels:
    # H4 = Close + Range * 1.1/2
    # H3 = Close + Range * 1.1/4
    # H2 = Close + Range * 1.1/6
    # H1 = Close + Range * 1.1/12
    # L1 = Close - Range * 1.1/12
    # L2 = Close - Range * 1.1/6
    # L3 = Close - Range * 1.1/4
    # L4 = Close - Range * 1.1/2
    
    range_1d = prev_day_high_6h - prev_day_low_6h
    close_1d = prev_day_close_6h
    
    # Calculate levels
    h4 = close_1d + range_1d * 1.1 / 2
    h3 = close_1d + range_1d * 1.1 / 4
    h2 = close_1d + range_1d * 1.1 / 6
    h1 = close_1d + range_1d * 1.1 / 12
    l1 = close_1d - range_1d * 1.1 / 12
    l2 = close_1d - range_1d * 1.1 / 6
    l3 = close_1d - range_1d * 1.1 / 4
    l4 = close_1d - range_1d * 1.1 / 2
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema_30_12h_aligned[i]) or np.isnan(h4[i]) or np.isnan(l4[i]) or
            np.isnan(vol_ma[i]) or np.isnan(prev_day_close_6h[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below H3 or trend turns bearish
            if close[i] < h3[i] or close[i] < ema_30_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above L3 or trend turns bullish
            if close[i] > l3[i] or close[i] > ema_30_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Fade at extreme levels in ranging market (price near H3/L3 but not beyond H4/L4)
                # Only fade if 12h EMA is flat (range-bound) - check if price is near EMA
                ema_distance = abs(close[i] - ema_30_12h_aligned[i]) / ema_30_12h_aligned[i]
                if ema_distance < 0.02:  # Price near 12h EMA suggests ranging market
                    # Fade at H3 (sell) or L3 (buy)
                    if close[i] > h3[i] and close[i] < h4[i]:
                        position = -1
                        signals[i] = -0.25
                    elif close[i] < l3[i] and close[i] > l4[i]:
                        position = 1
                        signals[i] = 0.25
                else:
                    # Trending market: breakout continuation
                    # Buy breakout above H4 in uptrend
                    if close[i] > h4[i] and close[i] > ema_30_12h_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                    # Sell breakdown below L4 in downtrend
                    elif close[i] < l4[i] and close[i] < ema_30_12h_aligned[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals