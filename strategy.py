#!/usr/bin/env python3
"""
1d_1w_momentum_range_v1
Hypothesis: Combine momentum and mean-reversion on daily timeframe with weekly trend filter.
- Momentum: Buy when RSI(14) crosses above 30 from below in bullish weekly trend (price > weekly EMA(50))
- Mean-reversion: Sell when RSI(14) crosses below 70 from above in bearish weekly trend (price < weekly EMA(50))
- Range filter: Avoid trading when Bollinger Band Width percentile < 20% (low volatility squeeze)
- Position sizing: 0.25 for long, -0.25 for short
- Target: 15-30 trades/year (60-120 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_momentum_range_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema_50_1w
    trend_1w_down = close_1w < ema_50_1w
    
    # Forward fill trend
    trend_1w_up_series = pd.Series(trend_1w_up)
    trend_1w_down_series = pd.Series(trend_1w_down)
    trend_1w_up_ffilled = trend_1w_up_series.ffill().values
    trend_1w_down_ffilled = trend_1w_down_series.ffill().values
    
    # Align 1w trend to 1d
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Bollinger Band Width for range filter
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean()
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # BBW percentile (252-day lookback for 1 year)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=252, min_periods=50).rank(pct=True) * 100
    bb_width_percentile = bb_width_percentile.fillna(50).values
    range_filter = bb_width_percentile >= 20  # Avoid low volatility squeeze
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(range_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 70 from above OR weekly trend turns down
            if i > 0 and rsi[i-1] >= 70 and rsi[i] < 70:
                position = 0
                signals[i] = 0.0
            elif trend_1w_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above 30 from below OR weekly trend turns up
            if i > 0 and rsi[i-1] <= 30 and rsi[i] > 30:
                position = 0
                signals[i] = 0.0
            elif trend_1w_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: RSI crosses above 30 from below + weekly uptrend + range filter
            if i > 0 and rsi[i-1] <= 30 and rsi[i] > 30 and trend_1w_up_aligned[i] and range_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: RSI crosses below 70 from above + weekly downtrend + range filter
            elif i > 0 and rsi[i-1] >= 70 and rsi[i] < 70 and trend_1w_down_aligned[i] and range_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals