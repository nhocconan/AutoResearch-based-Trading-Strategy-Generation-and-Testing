#!/usr/bin/env python3
name = "1d_Weekly_Engulfing_Trend_Strategy"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily volume spike detection (1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1d[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        # Bullish engulfing: current green candle fully engulfs previous red candle
        bullish_engulf = (close[i] > open_[i]) and (open_[i] < close[i-1]) and (close[i] > open_[i-1]) and (open_[i-1] > close[i-1])
        # Bearish engulfing: current red candle fully engulfs previous green candle
        bearish_engulf = (close[i] < open_[i]) and (open_[i] > close[i-1]) and (close[i] < open_[i-1]) and (open_[i-1] < close[i-1])
        
        if position == 0:
            # Long: bullish engulfing in weekly uptrend with volume
            if bullish_engulf and ema_20_1d[i] > ema_20_1d[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing in weekly downtrend with volume
            elif bearish_engulf and ema_20_1d[i] < ema_20_1d[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish engulfing or trend reversal
            if bearish_engulf or ema_20_1d[i] < ema_20_1d[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish engulfing or trend reversal
            if bullish_engulf or ema_20_1d[i] > ema_20_1d[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily engulfing patterns with weekly trend filter and volume confirmation
# - Engulfing candles signal strong momentum shifts at daily timeframe
# - Weekly EMA20 trend filter ensures we trade with the higher timeframe trend
# - Volume confirmation (1.5x average) reduces false signals
# - Works in both bull (engulfing in uptrend) and bear (engulfing in downtrend)
# - Target: ~15-25 trades/year to minimize fee drag
# - Uses 1d timeframe for signal generation and 1w for trend context
# - Engulfing patterns are reliable reversal/continuation signals in crypto markets
# - Simple 2-3 condition logic prevents overtrading and improves robustness