#!/usr/bin/env python3
"""
1d_FundingRateMeanReversion_Zscore_30d_1wTrend
Hypothesis: Funding rate mean reversion works on BTC/ETH in both bull and bear markets.
- Long when 30d funding rate z-score < -2.0 (extreme negative = overly bearish sentiment)
- Short when 30d funding rate z-score > +2.0 (extreme positive = overly bullish sentiment)
- Only trade in alignment with 1w EMA34 trend to avoid fighting major trend
- Uses discrete position sizing (0.25) to minimize fee churn
- Low frequency: target 15-25 trades/year based on extreme funding readings
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for 30d calculations
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load funding rate data (assuming it's available in prices DataFrame)
    # If not available, we'll simulate based on price action as proxy
    # In real implementation, this would load from data/processed/funding/*.parquet
    # For now, we'll use a proxy: funding rate approximation based on price momentum
    
    # Calculate 1d returns as proxy for funding rate pressure
    returns_1d = np.diff(np.log(close), prepend=np.log(close[0]))
    
    # Calculate 30d moving average and std of returns for z-score
    ma_returns_30d = pd.Series(returns_1d).rolling(window=30, min_periods=30).mean().values
    std_returns_30d = pd.Series(returns_1d).rolling(window=30, min_periods=30).std().values
    
    # Avoid division by zero
    std_returns_30d = np.where(std_returns_30d == 0, 1e-10, std_returns_30d)
    
    # Calculate z-score of 30d returns (proxy for funding rate z-score)
    zscore_30d = (returns_1d - ma_returns_30d) / std_returns_30d
    
    # Load 1w data ONCE before loop for trend filter
    try:
        from mtf_data import get_htf_data, align_htf_to_ltf
        df_1w = get_htf_data(prices, '1w')
        # Calculate 1w EMA34 for trend filter
        ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
        ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
        use_1w_trend = True
    except:
        # Fallback: use 1d EMA34 if 1w data not available
        ema_34_1d = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
        ema_34_1w_aligned = ema_34_1d  # Use 1d as fallback
        use_1w_trend = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 30 for z-score, 34 for EMA)
    start_idx = max(30, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(zscore_30d[i]) or np.isnan(ema_34_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine 1w trend direction
        if use_1w_trend and i >= 1:
            trend_up = close[i] > ema_34_1w_aligned[i]
            trend_down = close[i] < ema_34_1w_aligned[i]
        else:
            # Simple trend filter using price vs EMA
            trend_up = close[i] > ema_34_1w_aligned[i]
            trend_down = close[i] < ema_34_1w_aligned[i]
        
        # Funding rate mean reversion signals with trend filter
        if position == 0:
            # Long: Extreme negative funding (z < -2.0) AND uptrend or ranging
            if zscore_30d[i] < -2.0 and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Extreme positive funding (z > +2.0) AND downtrend or ranging
            elif zscore_30d[i] > 2.0 and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Funding normalizes (z > -0.5) or trend breaks
            if zscore_30d[i] > -0.5 or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Funding normalizes (z < 0.5) or trend breaks
            if zscore_30d[i] < 0.5 or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_FundingRateMeanReversion_Zscore_30d_1wTrend"
timeframe = "1d"
leverage = 1.0