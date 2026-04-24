#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and session filter.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA50 for trend filter (price above/below EMA50 defines bull/bear regime).
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-volume Asian session noise.
- Entry: Long when price breaks above Camarilla H3 in bull regime;
         Short when price breaks below Camarilla L3 in bear regime.
- Exit: Opposite Camarilla breakout (no trailing stop to reduce whipsaw).
- Signal size: 0.20 discrete to minimize fee churn and control drawdown.
- Designed for BTC/ETH: Camarilla levels provide institutional pivot points, EMA50 filter avoids counter-trend trades.
  Works in bull (breakouts with trend) and bear (strong moves after panic lows/highs).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels (H3, L3) on 1h data using current bar's OHLC
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_h3 = close + 1.1 * (high - low) / 2
    camarilla_l3 = close - 1.1 * (high - low) / 2
    # Shift to avoid look-ahead: levels calculated from current bar apply to next bar
    camarilla_h3 = np.roll(camarilla_h3, 1)
    camarilla_l3 = np.roll(camarilla_l3, 1)
    camarilla_h3[0] = np.nan
    camarilla_l3[0] = np.nan
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 1)  # EMA50 needs 50, plus 1 for roll
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Exit session filter: flatten position if outside session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price above/below 4h EMA50
        bull_regime = curr_close > ema_50_4h_aligned[i]
        bear_regime = curr_close < ema_50_4h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Camarilla H3 in bull regime
            if curr_close > camarilla_h3[i] and bull_regime:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla L3 in bear regime
            elif curr_close < camarilla_l3[i] and bear_regime:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: exit on opposite breakout (below L3)
            if curr_close < camarilla_l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit on opposite breakout (above H3)
            if curr_close > camarilla_h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50_Trend_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0