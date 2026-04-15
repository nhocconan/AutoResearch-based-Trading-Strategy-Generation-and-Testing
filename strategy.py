#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Width Regime + 1d ADX Trend Filter + Volume Spike
# Uses Bollinger Band Width (BBW) to identify range vs trend regimes.
# In trending regime (BBW > 50th percentile), trade breakouts in direction of 1d ADX trend.
# In ranging regime (BBW <= 50th percentile), fade moves to Bollinger Bands.
# Volume confirmation (>1.5x 20-bar median) filters low-quality signals.
# Designed to work in both bull (trend following) and bear (mean reversion) markets.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day ADX(14) for trend strength and direction
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_plus_ma = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_minus_ma = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_ma / (tr_ma + 1e-10)
    di_minus = 100 * dm_minus_ma / (tr_ma + 1e-10)
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_value = adx.values
    
    # ADX trend direction: +DI > -DI = up, else down
    adx_up = di_plus > di_minus
    
    # Align ADX and trend direction to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_value)
    adx_up_aligned = align_htf_to_ltf(prices, df_1d, adx_up.astype(float))
    
    # Bollinger Bands (20, 2) on 4h close
    bb_period = 20
    bb_mid = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    bb_std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid  # Normalized width
    
    # BBW percentile rank (lookback 50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_rank = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    )
    bb_rank = bb_rank.fillna(0.5).values  # Default to median if not enough data
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(adx_up_aligned[i]) or 
            np.isnan(bb_rank[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(vol_threshold[i])):
            continue
        
        # Regime: trending if BBW rank > 0.5, ranging if <= 0.5
        is_trending = bb_rank[i] > 0.5
        
        if is_trending:
            # Trend following: breakout in direction of 1d ADX trend
            if (adx_up_aligned[i] > 0.5 and  # ADX up
                close[i] > bb_upper[i] and   # Break above upper band
                volume[i] > vol_threshold[i]):  # Volume confirmation
                signals[i] = 0.25
            elif (adx_up_aligned[i] <= 0.5 and  # ADX down
                  close[i] < bb_lower[i] and    # Break below lower band
                  volume[i] > vol_threshold[i]):  # Volume confirmation
                signals[i] = -0.25
            else:
                # Hold or exit: reverse signal if opposite breakout
                if (i > 0 and signals[i-1] == 0.25 and 
                    close[i] < bb_lower[i] and volume[i] > vol_threshold[i]):
                    signals[i] = -0.25  # Reverse to short
                elif (i > 0 and signals[i-1] == -0.25 and 
                      close[i] > bb_upper[i] and volume[i] > vol_threshold[i]):
                    signals[i] = 0.25   # Reverse to long
                else:
                    signals[i] = signals[i-1]  # Hold position
        else:
            # Mean reversion: fade to Bollinger Band mean
            if (close[i] > bb_upper[i] and     # Sell at upper band
                volume[i] > vol_threshold[i]):
                signals[i] = -0.25
            elif (close[i] < bb_lower[i] and   # Buy at lower band
                  volume[i] > vol_threshold[i]):
                signals[i] = 0.25
            else:
                # Exit mean reversion when price returns to middle band
                if (i > 0 and signals[i-1] == 0.25 and 
                    close[i] <= bb_mid[i]):
                    signals[i] = 0.0
                elif (i > 0 and signals[i-1] == -0.25 and 
                      close[i] >= bb_mid[i]):
                    signals[i] = 0.0
                else:
                    signals[i] = signals[i-1]  # Hold position
    
    return signals

name = "4h_BBW_ADX_Volume_Regime"
timeframe = "4h"
leverage = 1.0