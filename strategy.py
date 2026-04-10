#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter
# - Long when price breaks above H3 (bullish bias) AND 4h EMA(21) > EMA(50) (bullish trend) AND UTC hour 8-20
# - Short when price breaks below L3 (bearish bias) AND 4h EMA(21) < EMA(50) (bearish trend) AND UTC hour 8-20
# - Exit when price returns to Pivot Point (mean reversion to equilibrium)
# - Uses discrete position sizing (0.20) to minimize fee churn
# - Camarilla pivots work well in ranging markets; 4h EMA filter prevents counter-trend trades in trends
# - Session filter (UTC 8-20) avoids low-liquidity Asian session noise
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Works in both bull and bear markets: mean reversion in ranges, trend filter prevents counter-trend trades

name = "1h_4h_camarilla_breakout_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h EMA trend filter: EMA(21) vs EMA(50)
    close_4h = df_4h['close'].values
    ema_21 = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_bullish = ema_21 > ema_50
    ema_bearish = ema_21 < ema_50
    
    # Align 4h EMA trend to 1h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_4h, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_4h, ema_bearish)
    
    # Pre-compute 1d Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    daily_range = high_1d - low_1d
    
    # Camarilla levels: H4, H3, H2, H1, Pivot, L1, L2, L3, L4
    pivot = (high_1d + low_1d + close_1d) / 3
    h3 = pivot + (1.1 * daily_range / 2)
    l3 = pivot - (1.1 * daily_range / 2)
    
    # Align 1d Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Pre-compute session filter (UTC 8-20)
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(in_session[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Get current price
        close_price = prices['close'].iloc[i]
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND 4h bullish trend AND in session
            if (close_price > h3_aligned[i] and 
                ema_bullish_aligned[i] and 
                in_session[i]):
                position = 1
                signals[i] = 0.20
            # Short when price breaks below L3 AND 4h bearish trend AND in session
            elif (close_price < l3_aligned[i] and 
                  ema_bearish_aligned[i] and 
                  in_session[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Pivot Point (mean reversion)
            # Exit when price returns to Pivot Point
            exit_long = position == 1 and close_price <= pivot_aligned[i]
            exit_short = position == -1 and close_price >= pivot_aligned[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals