#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for trend direction and 1h pullbacks to EMA21 for entry
# - Uses 4h Supertrend (ATR=10, mult=3) to determine trend direction (long only in uptrend, short only in downtrend)
# - Uses 1h EMA21 as dynamic support/resistance for pullback entries
# - Enters long when 4h Supertrend is uptrend and 1h price pulls back to touch EMA21 from above
# - Enters short when 4h Supertrend is downtrend and 1h price pulls back to touch EMA21 from below
# - Uses volume confirmation (1h volume > 1.5x 20-period average) to filter weak moves
# - Exits when price closes beyond EMA21 in the opposite direction or Supertrend flips
# - Designed to capture trend continuation moves with low frequency and high win rate
# - Target: 60-150 total trades over 4 years (15-37/year) with 0.20 position sizing
# - Works in bull markets by riding uptrends, works in bear markets by riding downtrends

name = "1h_4hSupertrend_EMA21_Pullback"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Supertrend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR for Supertrend
    def calculate_atr(high, low, close, period=10):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[period-1] = np.mean(tr[1:period+1])
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_4h = calculate_atr(high_4h, low_4h, close_4h, 10)
    
    # Calculate Supertrend
    def supertrend(high, low, close, atr, multiplier=3):
        hl2 = (high + low) / 2
        upper = hl2 + (multiplier * atr)
        lower = hl2 - (multiplier * atr)
        
        upper_band = np.zeros_like(close)
        lower_band = np.zeros_like(close)
        upper_band[0] = upper[0]
        lower_band[0] = lower[0]
        
        for i in range(1, len(close)):
            upper_band[i] = upper[i] if (upper[i] < upper_band[i-1] or close[i-1] > upper_band[i-1]) else upper_band[i-1]
            lower_band[i] = lower[i] if (lower[i] > lower_band[i-1] or close[i-1] < lower_band[i-1]) else lower_band[i-1]
        
        trend = np.ones_like(close)
        for i in range(1, len(close)):
            if close[i] > upper_band[i-1]:
                trend[i] = 1
            elif close[i] < lower_band[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
                if trend[i] == 1 and lower_band[i] < lower_band[i-1]:
                    lower_band[i] = lower_band[i-1]
                if trend[i] == -1 and upper_band[i] > upper_band[i-1]:
                    upper_band[i] = upper_band[i-1]
        
        return trend, upper_band, lower_band
    
    trend_4h, upper_band_4h, lower_band_4h = supertrend(high_4h, low_4h, close_4h, atr_4h, 3)
    
    # Align 4h Supertrend components to 1h
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    upper_band_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_band_4h)
    lower_band_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_band_4h)
    
    # 1h EMA21 for pullback entries
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation (1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(trend_4h_aligned[i]) or np.isnan(ema21[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h uptrend + price touches EMA21 from above + volume
            if (trend_4h_aligned[i] == 1 and 
                low[i] <= ema21[i] and 
                close[i] > ema21[i] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + price touches EMA21 from below + volume
            elif (trend_4h_aligned[i] == -1 and 
                  high[i] >= ema21[i] and 
                  close[i] < ema21[i] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below EMA21 OR 4h trend flips to downtrend
            if close[i] < ema21[i] or trend_4h_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above EMA21 OR 4h trend flips to uptrend
            if close[i] > ema21[i] or trend_4h_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals