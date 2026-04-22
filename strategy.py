#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h 15-period RSI + 4h EMA20 trend filter with volume spike
    # Targets 20-30 trades/year per symbol to minimize fee drag.
    # RSI mean reversion in ranging markets; EMA20 filters trend direction;
    # volume spike confirms institutional interest. Works in bull/bear via trend filter.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 15-period RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/15, adjust=False, min_periods=15).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/15, adjust=False, min_periods=15).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h EMA20 trend filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike filter (15-period)
    vol_ma15 = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    vol_spike = volume > 2.0 * vol_ma15  # Require 2x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema20[i]) or np.isnan(vol_ma15[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) + price above EMA20 (uptrend) + volume spike
            if rsi[i] < 30 and close[i] > ema20[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + price below EMA20 (downtrend) + volume spike
            elif rsi[i] > 70 and close[i] < ema20[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or trend reversal vs EMA20
            if position == 1:
                if rsi[i] > 50 or close[i] < ema20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi[i] < 50 or close[i] > ema20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_RSI15_EMA20_Volume_Spike_v1"
timeframe = "4h"
leverage = 1.0