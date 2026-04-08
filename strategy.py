#!/usr/bin/env python3
# 4h_adx_squeeze_breakout_1d_volume_v1
# Hypothesis: Bollinger Band squeeze breakout with ADX trend strength and volume confirmation on 4h.
# Long when BB width < 20th percentile (squeeze) and price breaks above upper band with volume > 2x average and ADX > 25.
# Short when BB width < 20th percentile and price breaks below lower band with volume > 2x average and ADX > 25.
# Exit when price returns to middle band (20-period SMA) or opposite signal.
# Designed to work in both bull and bear markets by capturing volatility breakouts from low volatility regimes.
# Target: 25-35 trades/year to minimize fee drag while capturing high-probability breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_adx_squeeze_breakout_1d_volume_v1"
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
    
    # Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    sma20 = close_series.rolling(window=20, min_periods=20).mean()
    std20 = close_series.rolling(window=20, min_periods=20).std()
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    middle_band = sma20
    bb_width = (upper_band - lower_band) / middle_band
    
    # BB width percentile (20-period lookback for regime)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    squeeze = bb_width_percentile < 20  # BB width in bottom 20% = squeeze
    
    # ADX (14) for trend strength
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.concatenate([[high[0]], high[:-1]])
    down_move = np.concatenate([[low[0]], low[:-1]]) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    strong_trend = adx > 25
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(squeeze[i]) or np.isnan(strong_trend[i]) or \
           np.isnan(avg_volume[i]) or np.isnan(upper_band[i]) or \
           np.isnan(lower_band[i]) or np.isnan(middle_band[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to middle band or opposite signal
            if close[i] <= middle_band[i] or \
               (close[i] < lower_band[i] and volume[i] > 2.0 * avg_volume[i] and strong_trend[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to middle band or opposite signal
            if close[i] >= middle_band[i] or \
               (close[i] > upper_band[i] and volume[i] > 2.0 * avg_volume[i] and strong_trend[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Squeeze condition: BB width in low volatility regime
            vol_cond = squeeze[i]
            # Volume confirmation: current volume > 2x average volume
            volume_ok = volume[i] > 2.0 * avg_volume[i]
            # Trend strength: ADX > 25
            trend_ok = strong_trend[i]
            
            # Long entry: price breaks above upper band with volume and trend strength
            if close[i] > upper_band[i] and vol_cond and volume_ok and trend_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower band with volume and trend strength
            elif close[i] < lower_band[i] and vol_cond and volume_ok and trend_ok:
                position = -1
                signals[i] = -0.25
    
    return signals