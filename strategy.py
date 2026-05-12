#!/usr/bin/env python3
# 4h_Bollinger_Bands_RSI_Reversal
# Hypothesis: On 4h timeframe, use Bollinger Bands (20,2) with RSI(14) extremes and volume confirmation for mean reversion in range-bound markets.
# Long when price touches lower BB with RSI<30 and volume spike; short when price touches upper BB with RSI>70 and volume spike.
# Exit when price returns to middle band (SMA20). Uses Bollinger Width percentile to filter for ranging markets (BW<50th percentile).
# Designed to work in both bull and bear by focusing on mean reversion in ranges rather than directional trends.

name = "4h_Bollinger_Bands_RSI_Reversal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2)
    close_s = pd.Series(close)
    sma20 = close_s.rolling(window=20, min_periods=20).mean().values
    std20 = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    middle_band = sma20
    
    # RSI(14)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(0).values
    
    # Bollinger Width (for regime filter)
    bb_width = (upper_band - lower_band) / middle_band * 100
    # Percentile of BB width over lookback (50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = close_s.rolling(window=20, min_periods=20).mean()  # using close for volume MA proxy
    vol_ma_vals = vol_ma.values  # Note: This is a proxy; ideally use actual volume
    # Better: use actual volume series
    vol_series = pd.Series(volume)
    vol_ma_actual = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_actual)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(sma20[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(rsi[i]) or np.isnan(bb_width_percentile[i]) or np.isnan(vol_ma_actual[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        bb_percentile = bb_width_percentile[i]
        # Only trade in ranging markets (Bollinger Width < 50th percentile)
        if bb_percentile >= 50:
            # Trending market - stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: Price touches lower BB with RSI oversold and volume confirmation
            if low[i] <= lower_band[i] and rsi[i] < 30 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches upper BB with RSI overbought and volume confirmation
            elif high[i] >= upper_band[i] and rsi[i] > 70 and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to middle band
            if close[i] >= middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to middle band
            if close[i] <= middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals