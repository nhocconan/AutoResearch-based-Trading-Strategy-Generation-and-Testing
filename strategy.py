#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Keltner Channel Breakout with Weekly Trend Filter
# Hypothesis: During low volatility (KC width < 25th percentile), price breaks out in direction of weekly EMA(20) trend.
# Works in bull/bear by trading breakouts with trend filter. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_keltner_breakout_weekly_trend_v1"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Keltner Channels (20, 1.5) on daily
    kc_period = 20
    kc_mult = 1.5
    ema_20 = pd.Series(close).ewm(span=kc_period, adjust=False).mean().values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=kc_period, adjust=False).mean().values
    
    upper = ema_20 + kc_mult * atr
    lower = ema_20 - kc_mult * atr
    kc_width = upper - lower
    
    # Keltner Channel width percentile (25-period lookback)
    kc_width_pct = pd.Series(kc_width).rolling(window=50, min_periods=25).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Weekly EMA(20) for trend filter
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_20[i]) or np.isnan(atr[i]) or np.isnan(kc_width_pct[i]) or 
            np.isnan(ema_20_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Low volatility condition: KC width in lowest 25% of recent range
        low_vol = kc_width_pct[i] <= 0.25
        
        if position == 1:  # Long position
            # Exit: price closes below EMA(20) or trend changes
            if close[i] < ema_20[i] or close[i] < ema_20_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above EMA(20) or trend changes
            if close[i] > ema_20[i] or close[i] > ema_20_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if low_vol:
                # Breakout above upper KC with uptrend
                if close[i] > upper[i] and close[i] > ema_20_weekly_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below lower KC with downtrend
                elif close[i] < lower[i] and close[i] < ema_20_weekly_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals