# 12h_price_action_reversal_v2
# Hypothesis: 12h price action reversals with volume confirmation and daily trend filter work in both bull and bear markets.
# Uses 12h price action reversal signals (engulfing candles, pin bars) with volume > 1.5x 20-period average.
# Daily EMA50 trend filter ensures alignment with higher timeframe trend.
# Target: 20-40 trades/year (80-160 over 4 years) with controlled risk.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_price_action_reversal_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 1.5
    
    # Daily trend filter: EMA50 on daily data
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Price action reversal signals
        body_size = abs(close[i] - open_price[i])
        total_range = high[i] - low[i]
        
        # Avoid division by zero
        if total_range == 0:
            signals[i] = 0.0
            continue
            
        # Bullish engulfing: current bullish candle engulfs previous bearish candle
        bullish_engulfing = (close[i] > open_price[i]) and (open_price[i-1] > close[i-1]) and \
                           (close[i] > open_price[i-1]) and (open_price[i] < close[i-1])
        
        # Bearish engulfing: current bearish candle engulfs previous bullish candle
        bearish_engulfing = (close[i] < open_price[i]) and (open_price[i-1] < close[i-1]) and \
                           (close[i] < open_price[i-1]) and (open_price[i] > close[i-1])
        
        # Bullish pin bar: long lower shadow, small upper shadow, body near top
        lower_shadow = min(open_price[i], close[i]) - low[i]
        upper_shadow = high[i] - max(open_price[i], close[i])
        bullish_pin = (lower_shadow > 2 * body_size) and (upper_shadow < 0.5 * body_size)
        
        # Bearish pin bar: long upper shadow, small lower shadow, body near bottom
        bearish_pin = (upper_shadow > 2 * body_size) and (lower_shadow < 0.5 * body_size)
        
        if position == 1:  # Long position
            # Exit: Bearish reversal signal or daily trend turns bearish
            if bearish_engulfing or bearish_pin or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bullish reversal signal or daily trend turns bullish
            if bullish_engulfing or bullish_pin or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: Bullish reversal with volume confirmation and daily uptrend
            if (bullish_engulfing or bullish_pin) and volume[i] > vol_threshold[i] and close[i] > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: Bearish reversal with volume confirmation and daily downtrend
            elif (bearish_engulfing or bearish_pin) and volume[i] > vol_threshold[i] and close[i] < ema_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals