#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h RSI Reversal with 4h Trend + Volume Filter
# Hypothesis: RSI mean reversion on 1h timeframe filtered by 4h trend direction
# and volume confirmation provides high-probability entries. In bull markets,
# we buy RSI pullbacks in uptrends; in bear markets, we sell RSI bounces in
# downtrends. Volume filter ensures momentum behind moves. Session filter
# (08-20 UTC) reduces noise. Target: 15-35 trades/year.
name = "1h_rsi_reversal_4h_trend_volume_v1"
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
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA(50) for trend
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or not session_mask[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend turns bearish
            if rsi[i] > 70 or close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend turns bullish
            if rsi[i] < 30 or close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation and session
            if vol_filter[i] and session_mask[i]:
                # Long setup: RSI < 30 (oversold) in uptrend
                if rsi[i] < 30 and close[i] > ema_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short setup: RSI > 70 (overbought) in downtrend
                elif rsi[i] > 70 and close[i] < ema_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals