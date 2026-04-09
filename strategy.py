#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h session-filtered Camarilla pivot breakout with 4h trend alignment and 1d volume confirmation
# Camarilla pivots provide precise intraday support/resistance levels for breakout/mean-reversion
# 4h EMA(50) determines primary trend direction - only take breakouts in trend direction
# 1d volume spike (volume > 1.5 * 20-period average) confirms institutional participation
# Session filter (08-20 UTC) reduces noise during low-liquidity Asian session
# Discrete position sizing 0.20 minimizes fee churn while maintaining adequate exposure
# Target: 60-120 total trades over 4 years (15-30/year) to stay within fee drag limits

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (avoid low-liquidity Asian session)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE for trend determination
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data ONCE for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period) for volume spike detection
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla pivot levels for current 1h bar
        # Using previous bar's OHLC (Camarilla uses previous day, but we adapt to 1h)
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        
        pivot = (phigh + plow + pclose) / 3.0
        range_val = phigh - plow
        
        # Camarilla levels
        h4 = pivot + (range_val * 1.1 / 2)
        h3 = pivot + (range_val * 1.1 / 4)
        h2 = pivot + (range_val * 1.1 / 6)
        h1 = pivot + (range_val * 1.1 / 12)
        l1 = pivot - (range_val * 1.1 / 12)
        l2 = pivot - (range_val * 1.1 / 6)
        l3 = pivot - (range_val * 1.1 / 4)
        l4 = pivot - (range_val * 1.1 / 2)
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Trend filter: price above/below 4h EMA(50)
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla H3 OR session ends
            if close[i] < h3 or not in_session[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla L3 OR session ends
            if close[i] > l3 or not in_session[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic with trend and volume confirmation
            if volume_confirmed:
                # In uptrend: buy break above H4 (bullish breakout)
                if uptrend and close[i] > h4:
                    position = 1
                    signals[i] = 0.20
                # In downtrend: sell break below L4 (bearish breakout)
                elif downtrend and close[i] < l4:
                    position = -1
                    signals[i] = -0.20
                # In ranging conditions (optional): mean reversion at H3/L3
                elif not uptrend and not downtrend:  # choppy/ranging market
                    if close[i] < l3 and volume_confirmed:
                        position = 1
                        signals[i] = 0.20
                    elif close[i] > h3 and volume_confirmed:
                        position = -1
                        signals[i] = -0.20
    
    return signals