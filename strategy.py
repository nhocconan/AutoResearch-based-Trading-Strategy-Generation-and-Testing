#!/usr/bin/env python3
# 1d_1w_price_action_reversal_v1
# Hypothesis: Daily price action reversals at weekly support/resistance levels with volume confirmation.
# Long when price bounces above weekly low with bullish engulfing candle and volume > 1.5x average.
# Short when price rejects below weekly high with bearish engulfing candle and volume > 1.5x average.
# Exit on opposite weekly level touch or engulfing reversal.
# Designed for low frequency (10-25 trades/year) to minimize fee drag and work in both bull/bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_price_action_reversal_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Align weekly levels to daily (wait for weekly close)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    
    # Volume filter: 1.5x 20-day average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Bullish engulfing: current candle engulfs previous bearish candle
    bullish_engulf = np.full(n, False)
    bearish_engulf = np.full(n, False)
    for i in range(1, n):
        # Bullish engulfing: current green candle fully engulfs previous red candle
        if (close[i] > open_price[i] and  # Current candle bullish
            open_price[i-1] > close[i-1] and  # Previous candle bearish
            close[i] >= open_price[i-1] and  # Current close >= previous open
            open_price[i] <= close[i-1]):  # Current open <= previous close
            bullish_engulf[i] = True
        # Bearish engulfing: current red candle fully engulfs previous bullish candle
        elif (open_price[i] > close[i] and  # Current candle bearish
              close[i-1] > open_price[i-1] and  # Previous candle bullish
              open_price[i] >= close[i-1] and  # Current open >= previous close
              close[i] <= open_price[i-1]):  # Current close <= previous open
            bearish_engulf[i] = True
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(vol_ma_period, 1) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price touches weekly high OR bearish engulfing signal
            if (high[i] >= weekly_high_aligned[i] or bearish_engulf[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price touches weekly low OR bullish engulfing signal
            if (low[i] <= weekly_low_aligned[i] or bullish_engulf[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price bounces above weekly low with bullish engulfing and volume surge
            if (low[i] <= weekly_low_aligned[i] * 1.001 and  # Touched or slightly below weekly low
                bullish_engulf[i] and
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price rejects below weekly high with bearish engulfing and volume surge
            elif (high[i] >= weekly_high_aligned[i] * 0.999 and  # Touched or slightly above weekly high
                  bearish_engulf[i] and
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals