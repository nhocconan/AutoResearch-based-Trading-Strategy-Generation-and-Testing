# 1d_Weekly_Trend_Follow_1wEMA13
# Hypothesis: Use weekly EMA13 as primary trend filter and daily price action for entries.
# Go long when daily close > weekly EMA13 and daily RSI(14) > 50 (bullish momentum).
# Go short when daily close < weekly EMA13 and daily RSI(14) < 50 (bearish momentum).
# Exit when RSI crosses back to 50 or weekly trend changes.
# This captures the major trend while avoiding counter-trend trades.
# Weekly trend filter reduces whipsaw, RSI provides timely entries/exits.
# Target: 15-25 trades/year (60-100 over 4 years) to minimize fee drag.

name = "1d_Weekly_Trend_Follow_1wEMA13"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Daily data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly EMA13 trend filter
    close_1w = df_1w['close'].values
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    
    # Daily RSI(14) for momentum and entry/exit
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # Neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 14  # for RSI
    
    for i in range(start_idx, n):
        # Skip if weekly EMA not available
        if np.isnan(ema13_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend direction
        weekly_uptrend = close[i] > ema13_1w_aligned[i]
        weekly_downtrend = close[i] < ema13_1w_aligned[i]
        
        # Daily RSI momentum
        rsi_now = rsi_values[i]
        rsi_bullish = rsi_now > 50
        rsi_bearish = rsi_now < 50
        
        if position == 0:
            # Look for entries aligned with weekly trend
            if weekly_uptrend and rsi_bullish:
                # Long: weekly uptrend + daily bullish momentum
                signals[i] = 0.25
                position = 1
            elif weekly_downtrend and rsi_bearish:
                # Short: weekly downtrend + daily bearish momentum
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: weekly trend turns down OR RSI turns bearish
                if not (weekly_uptrend and rsi_bullish):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: weekly trend turns up OR RSI turns bullish
                if not (weekly_downtrend and rsi_bearish):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals