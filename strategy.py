#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Bollinger Band Width + 1d Trend + Volume Confirmation
# Hypothesis: Bollinger Band Width (BBW) identifies low volatility regimes (squeeze) on 6h timeframe.
# In squeeze conditions (BBW < 20th percentile), we mean-revert at Bollinger Band edges.
# In expansion conditions (BBW > 80th percentile), we trend-follow with 1d EMA filter.
# Volume confirms institutional participation. This adapts to both ranging and trending markets.
# 6h timeframe balances responsiveness and noise reduction. Target: 12-37 trades/year (50-150 over 4 years).
name = "6b_bollinger_width_1d_trend_volume_v1"
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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Bollinger Bands on 6h timeframe (20, 2.0)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean()
    dev = close_s.rolling(window=20, min_periods=20).std()
    upper_band = basis + (dev * 2.0)
    lower_band = basis - (dev * 2.0)
    
    # Bollinger Band Width
    bb_width = (upper_band - lower_band) / basis
    # Percentile rank of BBW over 50 periods (adaptive regime detection)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # 1-day EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_6h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(bb_width_percentile[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(daily_ema_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches middle band (take profit) or breaks below lower band with volume
            if close[i] >= basis[i] or (close[i] < lower_band[i] and vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches middle band (take profit) or breaks above upper band with volume
            if close[i] <= basis[i] or (close[i] > upper_band[i] and vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Squeeze regime (low volatility): mean reversion at bands
                if bb_width_percentile[i] < 30:  # Low BBW = squeeze
                    # Long: price touches/bounces off lower band
                    if close[i] <= lower_band[i] * 1.001 and close[i] > lower_band[i] * 0.999:
                        position = 1
                        signals[i] = 0.25
                    # Short: price touches/bounces off upper band
                    elif close[i] >= upper_band[i] * 0.999 and close[i] <= upper_band[i] * 1.001:
                        position = -1
                        signals[i] = -0.25
                # Expansion regime (high volatility): trend following
                elif bb_width_percentile[i] > 70:  # High BBW = expansion
                    # Long: price above upper band with trend confirmation
                    if close[i] > upper_band[i] and close[i] > daily_ema_6h[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short: price below lower band with trend confirmation
                    elif close[i] < lower_band[i] and close[i] < daily_ema_6h[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals