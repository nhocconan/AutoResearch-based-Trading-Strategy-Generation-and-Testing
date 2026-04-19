# 12h_WeeklyDonchianBreakout_DailyTrend_v1
# Hypothesis: 12h strategy using weekly Donchian breakout (20 periods) with daily trend filter (EMA50) and volume confirmation.
# Weekly Donchian provides strong trend structure; daily EMA50 filters for higher timeframe trend alignment.
# Volume confirmation ensures breakouts have institutional participation.
# Works in bull markets via upward breakouts above weekly high + daily uptrend.
# Works in bear markets via downward breakouts below weekly low + daily downtrend.
# Target: 15-35 trades/year to stay well under fee drag limits.
name = "12h_WeeklyDonchianBreakout_DailyTrend_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # Get weekly data for Donchian channels (ONCE before loop)
    df_weekly = get_htf_data(prices, '1w')
    # Get daily data for EMA trend filter (ONCE before loop)
    df_daily = get_htf_data(prices, '1d')
    
    # Weekly Donchian channels (20-period high/low)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    donchian_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Daily EMA50 for trend filter
    close_daily = df_daily['close'].values
    ema50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # 12h ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(ema50_daily_aligned[i]) or np.isnan(atr_12h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_12h[i]
        
        # Volume filter: current volume > 1.8x average volume (30-period)
        if i >= 30:
            avg_volume = np.mean(volume[i-30:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.8 * avg_volume
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high + daily uptrend + volume
            if price > donchian_high_aligned[i] and price > ema50_daily_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low + daily downtrend + volume
            elif price < donchian_low_aligned[i] and price < ema50_daily_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses below weekly Donchian low or ATR stop
            if price < donchian_low_aligned[i] or price < close[i-1] - 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above weekly Donchian high or ATR stop
            if price > donchian_high_aligned[i] or price > close[i-1] + 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals