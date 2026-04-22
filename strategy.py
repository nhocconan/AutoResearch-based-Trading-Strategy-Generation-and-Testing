#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h momentum with 4h trend filter and volume confirmation
    # Uses 4h EMA20 for trend direction and 1h RSI(14) for entry timing.
    # Volume spike confirms institutional interest. Session filter (08-20 UTC) reduces noise.
    # Target: 15-35 trades/year to minimize fee drag while capturing trends in bull/bear.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # 1h RSI(14) for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(30, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI > 55 (bullish momentum) + price above 4h EMA20 (uptrend) + volume spike
            if rsi[i] > 55 and close[i] > ema20_4h_aligned[i] and vol_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI < 45 (bearish momentum) + price below 4h EMA20 (downtrend) + volume spike
            elif rsi[i] < 45 and close[i] < ema20_4h_aligned[i] and vol_spike[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI reverts to neutral (45-55) or trend reversal vs 4h EMA20
            if position == 1:
                if rsi[i] < 45 or close[i] < ema20_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if rsi[i] > 55 or close[i] > ema20_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI_EMA20_4h_Trend_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0