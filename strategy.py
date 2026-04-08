# 1d_Momentum_Trend_V1
# Hypothesis: Combines 1-week EMA trend filter with 1-day RSI momentum on pullbacks.
# In bull markets, buy RSI pullbacks above weekly EMA; in bear markets, sell RSI bounces below weekly EMA.
# This captures trend continuation with mean-reversion entries, reducing whipsaws.
# Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Momentum_Trend_V1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-week data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 21-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 14-period RSI on daily data
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = 21
    
    for i in range(start_idx, n):
        # Skip if weekly EMA is not available
        if np.isnan(ema_21_1w_aligned[i]):
            if position != 0:
                # Hold position until exit
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 50 (momentum fade) or weekly trend breaks
            if rsi_values[i] < 50 or close[i] < ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above 50 (momentum fade) or weekly trend breaks
            if rsi_values[i] > 50 or close[i] > ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: RSI pulls back to 30-40 and price above weekly EMA (bullish pullback)
            if 30 <= rsi_values[i] <= 40 and close[i] > ema_21_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: RSI bounces to 60-70 and price below weekly EMA (bearish bounce)
            elif 60 <= rsi_values[i] <= 70 and close[i] < ema_21_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals