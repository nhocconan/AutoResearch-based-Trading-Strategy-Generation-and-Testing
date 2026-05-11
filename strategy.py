#!/usr/bin/env python3
# 4h_ThreeWhiteSoldiers_BlackCrows_Volume
# Hypothesis: Captures trend reversals using Three White Soldiers/Black Crows candlestick patterns
# with volume confirmation and 12h EMA trend filter. Works in bull markets (catching bottoms) and
# bear markets (catching tops) by identifying exhaustion patterns. Uses 12h EMA50 for trend
# filtering to avoid counter-trend trades. Target: 20-50 trades/year to minimize fee drag.

name = "4h_ThreeWhiteSoldiers_BlackCrows_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h EMA50 for trend filter ---
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_prev = np.roll(ema_50_12h, 1)
    ema_50_12h_prev[0] = ema_50_12h[0]
    ema_50_12h_slope = ema_50_12h - ema_50_12h_prev
    ema_50_12h_slope = pd.Series(ema_50_12h_slope).ewm(span=3, adjust=False, min_periods=1).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_50_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_slope)
    
    # --- Three White Soldiers (bullish reversal) ---
    # Three consecutive bullish closes with higher closes
    bull1 = close > open_price
    bull2 = np.roll(close, 1) > np.roll(open_price, 1)
    bull3 = np.roll(close, 2) > np.roll(open_price, 2)
    higher_close1 = close > np.roll(close, 1)
    higher_close2 = np.roll(close, 1) > np.roll(close, 2)
    three_white_soldiers = bull1 & bull2 & bull3 & higher_close1 & higher_close2
    
    # --- Three Black Crows (bearish reversal) ---
    # Three consecutive bearish closes with lower closes
    bear1 = close < open_price
    bear2 = np.roll(close, 1) < np.roll(open_price, 1)
    bear3 = np.roll(close, 2) < np.roll(open_price, 2)
    lower_close1 = close < np.roll(close, 1)
    lower_close2 = np.roll(close, 1) < np.roll(close, 2)
    three_black_crows = bear1 & bear2 & bear3 & lower_close1 & lower_close2
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for pattern detection (3 bars) and EMA50 (50+3)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(ema_50_12h_slope_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 12h EMA50 slope
        uptrend = ema_50_12h_slope_aligned[i] > 0
        downtrend = ema_50_12h_slope_aligned[i] < 0
        
        if position == 0:
            # Long: Three White Soldiers + volume surge + in 12h uptrend or ranging
            if three_white_soldiers[i] and vol_surge[i]:
                # Allow long in uptrend or ranging (avoid strong downtrend)
                if not downtrend:  # not in strong 12h downtrend
                    signals[i] = 0.25
                    position = 1
            # Short: Three Black Crows + volume surge + in 12h downtrend or ranging
            elif three_black_crows[i] and vol_surge[i]:
                # Allow short in downtrend or ranging (avoid strong uptrend)
                if not uptrend:  # not in strong 12h uptrend
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit conditions for long
                # Exit on Three Black Crows (reversal signal) or strong 12h downtrend
                if three_black_crows[i] or downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit conditions for short
                # Exit on Three White Soldiers (reversal signal) or strong 12h uptrend
                if three_white_soldiers[i] or uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals