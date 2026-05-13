# 1h_HTF_Trend_With_Volume_Filter
# Hypothesis: Use 4h trend (EMA20) and 1d regime (ADX>25) for signal direction, with 1h entry on pullbacks to EMA20 (50) with volume confirmation.
# In trending markets (ADX>25), trade pullbacks to the 50 EMA in the direction of the 4h trend. In ranging markets (ADX<=25), stay flat.
# This reduces false signals and limits trades to ~20-40/year by requiring both trend alignment and volume confirmation.
# Works in bull markets (trend following) and bear markets (shorting trends) while avoiding chop.

name = "1h_HTF_Trend_With_Volume_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    ema20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Get 1d data for ADX (trend strength filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14)
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(0, high_1d[i] - high_1d[i-1])
        minus_dm[i] = max(0, low_1d[i-1] - low_1d[i])
        tr[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    tr_smooth = wilder_smooth(tr, period)
    
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = np.zeros_like(close_1d)
    dx[(plus_di + minus_di) != 0] = 100 * np.abs(plus_di[(plus_di + minus_di) != 0] - minus_di[(plus_di + minus_di) != 0]) / (plus_di[(plus_di + minus_di) != 0] + minus_di[(plus_di + minus_di) != 0])
    adx = wilder_smooth(dx, period)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1h EMA50 for entry timing
    ema50_1h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Only trade when ADX > 25 (trending market)
        if adx_aligned[i] > 25:
            # Determine trend direction from 4h EMA20
            uptrend = close[i] > ema20_4h_aligned[i]
            downtrend = close[i] < ema20_4h_aligned[i]
            
            if position == 0:
                # LONG: Pullback to EMA50 in uptrend with volume
                if uptrend and close[i] <= ema50_1h[i] * 1.005 and volume_filter[i]:
                    signals[i] = 0.20
                    position = 1
                # SHORT: Pullback to EMA50 in downtrend with volume
                elif downtrend and close[i] >= ema50_1h[i] * 0.995 and volume_filter[i]:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: Trend reversal or price moves above EMA50 significantly
                if not uptrend or close[i] > ema50_1h[i] * 1.02:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # EXIT SHORT: Trend reversal or price moves below EMA50 significantly
                if not downtrend or close[i] < ema50_1h[i] * 0.98:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
        else:
            # In ranging market (ADX <= 25), stay flat
            signals[i] = 0.0
            position = 0
    
    return signals