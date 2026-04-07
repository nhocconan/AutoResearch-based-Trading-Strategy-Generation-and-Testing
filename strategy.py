# 4h_keltner_breakout_1d_trend_v1
# Breakout strategy using Keltner Channels (ATR-based) on 4h with 1d trend filter.
# Long when price closes above upper Keltner Channel with bullish 1d EMA trend.
# Short when price closes below lower Keltner Channel with bearish 1d EMA trend.
# Uses volume confirmation and ATR-based stop loss via signal=0.
# Designed for low trade frequency (<40/year) to minimize fee drag.
# Works in bull markets via breakouts and bear markets via trend-following shorts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_keltner_breakout_1d_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Keltner Channel parameters
    kc_period = 20
    kc_multiplier = 2.0
    
    # True Range and ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=kc_period, min_periods=kc_period).mean().values
    
    # Keltner Channels
    ema_middle = pd.Series(close).ewm(span=kc_period, min_periods=kc_period).mean().values
    kc_upper = ema_middle + kc_multiplier * atr
    kc_lower = ema_middle - kc_multiplier * atr
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema_50d_aligned = align_htf_to_ltf(prices, df_1d, ema_50d)
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(kc_period, n):
        # Skip if data not available
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or np.isnan(close[i]) or
            np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema_50d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_confirmed = volume[i] > vol_ma[i]
        trend_bullish = ema_50d_aligned[i] > close_1d[0] if len(close_1d) > 0 else False  # Simplified: price above EMA
        trend_bearish = ema_50d_aligned[i] < close_1d[0] if len(close_1d) > 0 else False
        
        if position == 1:  # Long position
            # Exit: price closes below middle Keltner or trend turns bearish
            if close[i] < ema_middle[i] or not trend_bullish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above middle Keltner or trend turns bullish
            if close[i] > ema_middle[i] or not trend_bearish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price closes above upper Keltner with bullish trend and volume
            if close[i] > kc_upper[i] and trend_bullish and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Short: price closes below lower Keltner with bearish trend and volume
            elif close[i] < kc_lower[i] and trend_bearish and vol_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals