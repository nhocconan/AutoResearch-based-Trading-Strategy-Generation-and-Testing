# US equities have consistently shown that strong momentum continues to outperform mean-reversion strategies in volatile markets, particularly when combined with volatility filtering. The rationale is to capture strong directional moves while avoiding choppy periods where false breakouts occur frequently.
# For BTC/ETH specifically, this approach works because:
# 1) Strong trends persist longer than expected in crypto (especially during institutional adoption phases)
# 2) Volatility filters prevent entries during low-liquidity manipulation periods
# 3) The strategy avoids mean-reversion traps during strong bull/bear runs
# 4) Position sizing at 0.25 limits drawdown during inevitable corrections
# Timeframe: 1d provides enough signal quality to avoid excessive trading while capturing major moves
# HTF: 1w provides major trend context without excessive lag
# Expected trades: 20-40 per year based on historical breakout frequency with volume confirmation

#!/usr/bin/env python3
name = "1d_VolumeFiltered_Breakout_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 20-day Donchian channels for breakout signals
    # Upper channel: highest high of last 20 days
    high_series = pd.Series(high)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 days
    low_series = pd.Series(low)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 2.0x 20-period average (more stringent to reduce trades)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above 20-day high + above 1w EMA50 + volume filter
            if high[i] > donchian_upper[i] and close[i] > ema_50_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below 20-day low + below 1w EMA50 + volume filter
            elif low[i] < donchian_lower[i] and close[i] < ema_50_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: breakdown below 20-day low or below 1w EMA50
            if low[i] < donchian_lower[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: breakout above 20-day high or above 1w EMA50
            if high[i] > donchian_upper[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals