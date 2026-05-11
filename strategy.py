# Hypothesis: 1d Bollinger Band squeeze breakout with weekly trend and volume confirmation
# Works in bull markets (breakouts with volume) and bear (mean reversion from squeeze in ranging markets)
# Uses squeeze as low-volatility precursor to explosive moves, filtered by weekly trend direction
# Target: 15-25 trades/year on BTC/ETH to avoid fee drag

#!/usr/bin/env python3
name = "1d_BollingerSqueeze_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_mid + bb_std_dev * bb_std
    bb_lower = bb_mid - bb_std_dev * bb_std
    
    # Bollinger Band Width (normalized)
    bb_width = (bb_upper - bb_lower) / bb_mid
    # Percentile of BB width over 250 days (~1 year)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=250, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Squeeze condition: BB width in lowest 10% percentile (low volatility)
    squeeze_condition = bb_width_percentile < 10
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly 50-period EMA for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(bb_mid[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma20[i]) or np.isnan(bb_width_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Squeeze breakout above upper BB + weekly uptrend + volume spike
            if (squeeze_condition[i] and close[i] > bb_upper[i] and 
                close[i] > ema_50_1w_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Squeeze breakout below lower BB + weekly downtrend + volume spike
            elif (squeeze_condition[i] and close[i] < bb_lower[i] and 
                  close[i] < ema_50_1w_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price closes below middle BB OR weekly trend turns down
            if close[i] < bb_mid[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price closes above middle BB OR weekly trend turns up
            if close[i] > bb_mid[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals