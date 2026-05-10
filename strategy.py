# WARNING: This is a template for educational purposes only.
# Implementing this code may result in losses.
# The author assumes no liability for your use of this code.
# Always backtest and consult a financial professional before live trading.

#!/usr/bin/env python3
"""
4h_Engulfing_Signal_With_Volume_Confirmation
Hypothesis: Bullish/bearish engulfing candles on 4h chart, confirmed by volume spike and 1d EMA trend filter, provide high-probability entries in both bull and bear markets.
Engulfing candles signal strong momentum shifts; volume confirms institutional interest; EMA filter avoids counter-trend trades.
Designed for ~20-40 trades/year to minimize fee drag while capturing significant moves.
"""

name = "4h_Engulfing_Signal_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 4h price and volume
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    # Bullish engulfing: current green candle fully engulfs previous red candle
    bullish_engulf = (close > open_price) & (open_price < close) & (close > open_price) & \
                     (open_price < close) & (close > open_price) & (open_price < close) & \
                     (close > open_price) & (open_price < close)  # Placeholder - actual logic below
    
    # Actually: bullish engulfing = current candle is green (close > open) AND
    # its body completely engulfs previous candle's body (which was red)
    bullish_engulf = (close > open_price) & (open_price < close) & \
                     (close > open_price) & (open_price < close) & \
                     (close[1:] > open_price[:-1]) & (open_price[1:] < close[:-1])  # Still wrong
    
    # Correct implementation:
    bullish_engulf = (close > open_price) & (open_price < close)  # Temp
    
    # Proper bullish engulfing: current green candle body > previous red candle body
    # and engulfs it completely
    curr_body = close - open_price
    prev_body = open_price - close  # Will be negative for red candle
    bullish_engulf = (close > open_price) & (open_price < close) & \
                     (curr_body > -prev_body) & (open_price <= close) & (close >= open_price)
    # Actually simpler:
    bullish_engulf = (close > open_price) & (open_price < close) & \
                     (close > open_price) & (open_price < close)  # Reset
    
    # Correct bullish engulfing:
    bullish_engulf = (close > open_price) & (open_price < close)  # Current green
    prev_red = (open_price > close)  # Previous red (actually need to shift)
    prev_red = np.concatenate([[False], open_price[:-1] > close[:-1]])  # Previous bar red
    bullish_engulf = bullish_engulf & prev_red & (close > open_price) & (open_price < close)  # Still messy
    
    # Let's do it properly:
    is_green = close > open_price
    is_red = open_price > close
    # Bullish engulf: current green, previous red, and current close > previous open AND current open < previous close
    bullish_engulf = is_green & np.concatenate([[False], is_red[:-1]]) & \
                     (close > np.concatenate([[open_price[0]], open_price[:-1]])) & \
                     (open_price < np.concatenate([[close[0]], close[:-1]]))
    
    # Bearish engulf: current red, previous green, and current open > previous close AND current close < previous open
    bearish_engulf = is_red & np.concatenate([[False], is_green[:-1]]) & \
                     (open_price > np.concatenate([[close[0]], close[:-1]])) & \
                     (close < np.concatenate([[open_price[0]], open_price[:-1]]))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50 days) + engulfing needs 1 previous bar
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish engulfing AND above 1d EMA50 (uptrend) AND volume filter
            if bullish_engulf[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing AND below 1d EMA50 (downtrend) AND volume filter
            elif bearish_engulf[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish engulfing OR price closes below 1d EMA50
            if bearish_engulf[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish engulfing OR price closes above 1d EMA50
            if bullish_engulf[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals