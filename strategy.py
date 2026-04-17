# 4h_PriceAction_Confirmation_Strategy
# Hypothesis: Price action confirmation using bullish/bearish engulfing patterns with volume and trend filter on 4h timeframe. Works in bull/bear by capturing momentum shifts at key levels with volume confirmation.
# Timeframe: 4h, uses 1d for trend filter. Target: 30-60 trades/year to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average (4h bars)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulf = (close > open_) & (open_ < close) & (close_[1] < open_[1]) & (close > open_[1]) & (open_ < close_[1])
    # Bearish engulfing: current red candle engulfs previous green candle
    bearish_engulf = (close < open_) & (open_ > close) & (close_[1] > open_[1]) & (close < open_[1]) & (open_ > close_[1])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above EMA34 for long, below for short
        trend_filter_long = close[i] > ema34_1d_aligned[i]
        trend_filter_short = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long entry: bullish engulfing with volume and trend filter
            if i >= 1 and bullish_engulf[i] and volume_filter and trend_filter_long:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish engulfing with volume and trend filter
            elif i >= 1 and bearish_engulf[i] and volume_filter and trend_filter_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish engulfing or trend reversal
            if i >= 1 and bearish_engulf[i] and volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish engulfing or trend reversal
            if i >= 1 and bullish_engulf[i] and volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PriceAction_Confirmation_Strategy"
timeframe = "4h"
leverage = 1.0