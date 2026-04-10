#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume spike and 1w trend filter
# - Primary: 4h Williams %R(14) below -80 for oversold long, above -20 for overbought short
# - HTF volume filter: 1d volume > 1.5x 20-period MA for institutional participation
# - HTF trend filter: 1w close > 1w EMA50 for long bias, < EMA50 for short bias
# - Entry: Long when Williams %R < -80 + volume filter + 1w uptrend; Short when Williams %R > -20 + volume filter + 1w downtrend
# - Exit: Williams %R crosses above -50 for long exit, below -50 for short exit
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# - Works in bull/bear: Williams %R captures mean reversion in ranging markets, volume confirms validity, 1w trend ensures alignment with higher timeframe momentum

name = "4h_1d_1w_williamsr_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R(14) on 4h
    def calculate_williams_r(high, low, close, lookback=14):
        highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
        lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero when highest_high == lowest_low
        wr = np.where((highest_high - lowest_low) == 0, -50, wr)
        return wr
    
    wr_4h = calculate_williams_r(high, low, close, 14)
    
    # Calculate 1d volume MA(20)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(60, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(wr_4h[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        # Trend filter: 1w close > EMA50 for uptrend, < EMA50 for downtrend
        trend_up = close_1w[-1] > ema_50_1w[-1] if len(close_1w) > 0 else False
        trend_down = close_1w[-1] < ema_50_1w[-1] if len(close_1w) > 0 else False
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) + volume confirmation + 1w uptrend
            if (wr_4h[i] < -80 and volume_confirm and trend_up):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R > -20 (overbought) + volume confirmation + 1w downtrend
            elif (wr_4h[i] > -20 and volume_confirm and trend_down):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R crosses above -50 for long, below -50 for short
            if position == 1:  # Long position
                if wr_4h[i] > -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if wr_4h[i] < -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals