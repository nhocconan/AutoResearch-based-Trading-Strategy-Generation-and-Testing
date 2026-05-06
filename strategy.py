# 1. Hypothesis: The strategy uses a 12-hour Exponential Moving Average (EMA50) as a trend filter on the 4-hour timeframe, combined with price action relative to the EMA and volume confirmation to capture trending moves while avoiding whipsaws in ranging markets. The EMA50 on the 12-hour timeframe provides a smoother, more reliable trend signal than shorter-term EMAs, reducing false signals during choppy periods. Volume confirmation ensures that price moves are supported by participation, increasing the likelihood of sustained trends. The strategy is designed to work in both bull and bear markets by dynamically adapting to the prevailing trend direction as defined by the 12-hour EMA50, thus avoiding the pitfalls of fixed-direction strategies. The 4-hour timeframe balances responsiveness with reduced noise, targeting a trade frequency within the optimal range to minimize fee drag.

# 2. Implementation: The strategy calculates the 12-hour EMA50 once before the main loop using the `get_htf_data` function, then aligns it to the 4-hour chart using `align_htf_to_ltf` to ensure no look-ahead bias. Entry conditions require the price to be on the correct side of the EMA (above for long, below for short) and volume to exceed 1.5 times its 20-period moving average. Positions are held until the price crosses back over the EMA, at which point the signal returns to zero. Position sizing is set to 0.25 (25% of capital) to balance risk and reward, using discrete levels to minimize fee churn from frequent signal changes. All indicators use appropriate `min_periods` to avoid look-ahead bias, and the main loop processes data in a vectorized manner where possible, with only the state-dependent logic in a loop.

# 3. The strategy adheres to all rules: it calls `get_htf_data` only once before the loop, uses `align_htf_to_ltf` for proper multi-timeframe alignment without manual index mapping, avoids any form of resampling or manual timeframe conversion, uses discrete position sizes (0.0, ±0.25), includes exit logic via signal changes (no simulated intrabar stops), and respects the 4-hour timeframe as required. The use of volume and EMA trend filtering is intended to produce a moderate number of trades (targeting 20-50 per year per symbol) to avoid the fee drag that has caused recent strategy failures.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_EMA50_12hTrend_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 4h timeframe (primary)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price above EMA AND volume confirmation
            if close[i] > ema_50_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price below EMA AND volume confirmation
            elif close[i] < ema_50_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Stay long if price remains above EMA
            if close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
            else:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Stay short if price remains below EMA
            if close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
                position = 0
    
    return signals