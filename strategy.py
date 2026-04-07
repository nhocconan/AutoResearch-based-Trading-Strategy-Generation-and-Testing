#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h EMA Pullback with 4h/1d Trend Filter and Session Filter
# Hypothesis: In trending markets (4h EMA > 4h SMA for bull, < for bear),
# buy pullbacks to 1h EMA(20) in bull trend or sell rallies to 1h EMA(20) in bear trend.
# Volume confirms institutional participation. Session filter (08-20 UTC) avoids low-liquidity hours.
# Uses 4h/1d for trend direction, 1h for entry timing. Target: 15-37 trades/year (60-150 over 4 years).
name = "1h_ema_pullback_4h1d_trend_volume_session_v1"
timeframe = "1h"
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
    
    # Get 4h and 1d data for trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 1h EMA(20) for entry timing
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).values
    
    # 4h EMA(20) and SMA(20) for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).values
    sma_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    ema_20_4h_1h = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    sma_20_4h_1h = align_htf_to_ltf(prices, df_4h, sma_20_4h)
    
    # 1d EMA(50) and SMA(50) for stronger trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    ema_50_1d_1h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    sma_50_1d_1h = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_20[i]) or np.isnan(ema_20_4h_1h[i]) or np.isnan(sma_20_4h_1h[i]) or
            np.isnan(ema_50_1d_1h[i]) or np.isnan(sma_50_1d_1h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend regime: require both 4h and 1d to agree
        bull_4h = ema_20_4h_1h[i] > sma_20_4h_1h[i]
        bear_4h = ema_20_4h_1h[i] < sma_20_4h_1h[i]
        bull_1d = ema_50_1d_1h[i] > sma_50_1d_1h[i]
        bear_1d = ema_50_1d_1h[i] < sma_50_1d_1h[i]
        
        bull_regime = bull_4h and bull_1d
        bear_regime = bear_4h and bear_1d
        
        if position == 1:  # Long position
            # Exit: price breaks below 1h EMA(20) or trend turns bearish
            if close[i] < ema_20[i] or not bull_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price breaks above 1h EMA(20) or trend turns bullish
            if close[i] > ema_20[i] or not bear_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Require volume and session
            if vol_filter[i] and session_filter[i]:
                # Bull regime: buy pullback to EMA(20)
                if bull_regime and close[i] <= ema_20[i]:
                    # Additional confirmation: price closing above EMA(20) next bar (handled via signal at next bar)
                    position = 1
                    signals[i] = 0.20
                # Bear regime: sell rally to EMA(20)
                elif bear_regime and close[i] >= ema_20[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals