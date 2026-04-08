#!/usr/bin/env python3
# 1d_donchian_breakout_1w_trend_volume_v2
# Hypothesis: Daily Donchian(20) breakout with weekly trend filter (EMA50) and volume confirmation.
# Long when price breaks above 20-day high with volume > 1.5x average and weekly EMA50 rising.
# Short when price breaks below 20-day low with volume > 1.5x average and weekly EMA50 falling.
# Exit when price returns to 10-day moving average or opposite signal.
# Designed to capture medium-term trends while avoiding false breakouts in low volume conditions.
# Target: 10-20 trades/year to minimize fee decay while capturing strong momentum moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # Daily 10-period moving average for exit
    ma10 = pd.Series(close).rolling(window=10, min_periods=10).mean()
    
    # Weekly trend filter: EMA50
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    # Weekly EMA slope (rising/falling)
    ema50_series = pd.Series(weekly_ema50_aligned)
    ema50_slope = ema50_series.diff(5)  # 5-day slope
    weekly_uptrend = ema50_slope > 0
    weekly_downtrend = ema50_slope < 0
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ma10[i]) or np.isnan(weekly_ema50_aligned[i]) or \
           np.isnan(weekly_uptrend[i]) or np.isnan(weekly_downtrend[i]) or \
           np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to 10-day MA or opposite signal
            if close[i] <= ma10[i] or \
               (close[i] < donchian_low[i] and volume[i] > 1.5 * avg_volume[i] and weekly_downtrend[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to 10-day MA or opposite signal
            if close[i] >= ma10[i] or \
               (close[i] > donchian_high[i] and volume[i] > 1.5 * avg_volume[i] and weekly_uptrend[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price breaks above 20-day high with volume and weekly uptrend
            if close[i] > donchian_high[i] and volume_ok and weekly_uptrend[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-day low with volume and weekly downtrend
            elif close[i] < donchian_low[i] and volume_ok and weekly_downtrend[i]:
                position = -1
                signals[i] = -0.25
    
    return signals