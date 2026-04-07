#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Bollinger Bands with 1d Trend Filter and Volume Confirmation
# Hypothesis: Bollinger Bands identify mean reversion opportunities in ranging markets.
# We trade reversals from the bands when aligned with the daily trend (EMA50) and volume spikes.
# This strategy works in both bull and bear markets by trading mean reversion within the trend.
# Target: 20-50 trades/year (80-200 over 4 years) to minimize fee drag.
name = "4h_bollinger_reversion_1d_trend_volume_v1"
timeframe = "4h"
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
    
    # Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    bb_basis = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    bb_dev = bb_mult * pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev
    
    # 1-day EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_4h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(bb_length, n):
        # Skip if required data not available
        if (np.isnan(bb_basis[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(daily_ema_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to middle band or trend turns bearish
            if close[i] >= bb_basis[i] or close[i] < daily_ema_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price returns to middle band or trend turns bullish
            if close[i] <= bb_basis[i] or close[i] > daily_ema_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long: price at or below lower band AND above daily EMA (uptrend)
                if close[i] <= bb_lower[i] and close[i] > daily_ema_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price at or above upper band AND below daily EMA (downtrend)
                elif close[i] >= bb_upper[i] and close[i] < daily_ema_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals