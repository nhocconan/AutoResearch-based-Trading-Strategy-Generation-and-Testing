#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray Power + 1w Trend + Volume Confirmation
# Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength.
# In weak trends (Elder Ray near zero), we mean-revert at EMA13 crossovers with volume.
# In strong trends (|Elder Ray| > threshold), we trend-follow with 1-week EMA filter.
# Volume confirms institutional participation. Works in bull/bear via regime adaptation.
# 6h timeframe reduces noise. Target: 12-37 trades/year (50-150 over 4 years).
name = "6h_elder_ray_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Elder Ray components on 6h timeframe (EMA13)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high - ema13.values  # High - EMA13
    bear_power = ema13.values - low   # EMA13 - Low
    elder_ray = bull_power - bear_power  # Net strength: + = bullish, - = bearish
    
    # Absolute Elder Ray for trend strength detection
    abs_elder_ray = np.abs(elder_ray)
    # Percentile rank of |Elder Ray| over 50 periods (adaptive regime detection)
    abs_er_series = pd.Series(abs_elder_ray)
    abs_er_percentile = abs_er_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # 1-week EMA(20) for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_6h = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(abs_er_percentile[i]) or np.isnan(ema13.iloc[i]) or 
            np.isnan(weekly_ema_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Elder Ray turns negative (bearish) OR price crosses below EMA13 with volume
            if elder_ray[i] < 0 or (close[i] < ema13.iloc[i] and vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Elder Ray turns positive (bullish) OR price crosses above EMA13 with volume
            if elder_ray[i] > 0 or (close[i] > ema13.iloc[i] and vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Weak trend regime (low |Elder Ray|): mean reversion at EMA13
                if abs_er_percentile[i] < 40:  # Low |Elder Ray| = weak trend
                    # Long: price crosses above EMA13 from below
                    if close[i] > ema13.iloc[i] and close[i-1] <= ema13.iloc[i-1]:
                        position = 1
                        signals[i] = 0.25
                    # Short: price crosses below EMA13 from above
                    elif close[i] < ema13.iloc[i] and close[i-1] >= ema13.iloc[i-1]:
                        position = -1
                        signals[i] = -0.25
                # Strong trend regime (high |Elder Ray|): trend following
                elif abs_er_percentile[i] > 60:  # High |Elder Ray| = strong trend
                    # Long: Elder Ray positive AND price above weekly EMA
                    if elder_ray[i] > 0 and close[i] > weekly_ema_6h[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short: Elder Ray negative AND price below weekly EMA
                    elif elder_ray[i] < 0 and close[i] < weekly_ema_6h[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals