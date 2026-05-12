# 6h_BollingerBreakout_1wTrend_PriceAction
# Bollinger Band breakout with weekly trend filter and price action confirmation
# Works in bull/bear by combining volatility breakout with higher timeframe trend
# Target: 50-150 total trades over 4 years (12-37/year)
# Size: 0.25-0.30

#!/usr/bin/env python3
name = "6h_BollingerBreakout_1wTrend_PriceAction"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtrader import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Bollinger Bands (20, 2.0)
    bb_period = 20
    bb_std = 2.0
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_bb + (bb_std * bb_std_dev)
    bb_lower = sma_bb - (bb_std * bb_std_dev)
    
    # Price action: close must be outside Bollinger Band for 2 consecutive closes
    bb_breakout_up = (close > bb_upper) & (np.roll(close, 1) > np.roll(bb_upper, 1))
    bb_breakout_down = (close < bb_lower) & (np.roll(close, 1) < np.roll(bb_lower, 1))
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, bb_period)  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(sma_bb[i]) or np.isnan(bb_std_dev[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bollinger breakout up + above weekly EMA34 + volume filter
            if bb_breakout_up[i] and close[i] > ema_34_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bollinger breakout down + below weekly EMA34 + volume filter
            elif bb_breakout_down[i] and close[i] < ema_34_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bollinger breakout down OR below weekly EMA34
            if bb_breakout_down[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bollinger breakout up OR above weekly EMA34
            if bb_breakout_up[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals