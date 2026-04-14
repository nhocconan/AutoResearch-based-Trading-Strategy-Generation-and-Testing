# 1h_4d_Momentum_Reversal_Scalper
# Hypothesis: In BTC/ETH, intraday momentum reversals occur at key levels (pivot S3/R3) when volume confirms rejection.
# Uses 1d pivot levels (S3/R3) for direction, 1h for entry timing. Filters: volume > 20MA, price vs EMA50(1d) for trend.
# Works in bull (buying dips) and bear (selling rallies). Target: 15-35 trades/year via strict entry conditions.
# Timeframe: 1h, leverage: 1.0, position size: 0.20

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 50:
        ema_series = pd.Series(close_1d)
        ema50_1d = ema_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA50 to 1h timeframe
    ema50_1h_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1h volume moving average (20-period) for volume filter
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        volume_series = pd.Series(volume)
        volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if np.isnan(ema50_1h_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        # Get previous day's data (1d index) - safe for 1h timeframe
        if i >= 1:
            prev_close = close_1d[i-1]
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            
            # Calculate pivot points (standard formula)
            pivot = (prev_high + prev_low + prev_close) / 3.0
            s1 = (2 * pivot) - prev_high
            r1 = (2 * pivot) - prev_low
            s2 = pivot - (prev_high - prev_low)
            r2 = pivot + (prev_high - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot)
            r3 = prev_high + 2 * (pivot - prev_low)
            
            # Align S3/R3 to 1h timeframe (constant values for the day)
            s3_array = np.full(len(df_1d), s3)
            r3_array = np.full(len(df_1d), r3)
            s3_1h = align_htf_to_ltf(prices, df_1d, s3_array)[i]
            r3_1h = align_htf_to_ltf(prices, df_1d, r3_array)[i]
            
            if position == 0:
                # Long: Price rejects S3 with volume and above EMA50 (bullish trend)
                if low[i] <= s3_1h and close[i] > s3_1h and volume[i] > volume_ma[i] and close[i] > ema50_1h_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: Price rejects R3 with volume and below EMA50 (bearish trend)
                elif high[i] >= r3_1h and close[i] < r3_1h and volume[i] > volume_ma[i] and close[i] < ema50_1h_aligned[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit: Price breaks S3 again or trend changes (price below EMA50)
                if low[i] <= s3_1h or close[i] < ema50_1h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price breaks R3 again or trend changes (price above EMA50)
                if high[i] >= r3_1h or close[i] > ema50_1h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4d_Momentum_Reversal_Scalper"
timeframe = "1h"
leverage = 1.0