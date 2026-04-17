# %%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA(21) + 4h EMA(50) trend filter + 1d volume spike + session filter (08-20 UTC).
# Long when price > EMA21(1h) AND EMA50(4h) rising AND volume > 2x 20-period average (1d) AND in session.
# Short when price < EMA21(1h) AND EMA50(4h) falling AND volume > 2x 20-period average (1d) AND in session.
# Uses higher timeframe for trend direction and volume confirmation, lower timeframe for entry timing.
# Session filter reduces noise trades outside active hours. Designed for low turnover (target: 15-35 trades/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA21 on 1h close
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate EMA50 on 4h close
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate EMA50 slope on 4h (rising/falling)
    ema50_slope = np.zeros_like(ema50_4h_aligned)
    ema50_slope[1:] = ema50_4h_aligned[1:] - ema50_4h_aligned[:-1]
    
    # Calculate 20-period average volume on 1d
    volume_1d_series = pd.Series(volume_1d)
    volume_ma20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for EMA calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21[i]) or 
            np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema50_slope[i]) or 
            np.isnan(volume_ma20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1h volume > 2 * 20-period average volume (1d aligned)
        volume_filter = volume[i] > (2.0 * volume_ma20_1d_aligned[i])
        
        if position == 0:
            # Long: price > EMA21 AND EMA50 rising AND volume filter AND in session
            if (close[i] > ema21[i] and 
                ema50_slope[i] > 0 and 
                volume_filter and 
                in_session[i]):
                signals[i] = 0.20
                position = 1
            # Short: price < EMA21 AND EMA50 falling AND volume filter AND in session
            elif (close[i] < ema21[i] and 
                  ema50_slope[i] < 0 and 
                  volume_filter and 
                  in_session[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA21 OR EMA50 slope turns negative
            if close[i] < ema21[i] or ema50_slope[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above EMA21 OR EMA50 slope turns positive
            if close[i] > ema21[i] or ema50_slope[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA21_4hEMA50_Trend_Volume_Session"
timeframe = "1h"
leverage = 1.0
# %%