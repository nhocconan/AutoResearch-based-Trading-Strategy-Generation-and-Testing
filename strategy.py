#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA trend filter + volume confirmation
# - Primary: 1d Donchian breakout (20-day high/low) for trend continuation
# - HTF: 1w EMA(21) for long-term trend filter (bullish above EMA, bearish below)
# - Volume: 1d volume > 1.5x 20-day average for conviction
# - Long: Price breaks above 20-day high + close > 1w EMA + volume confirmation
# - Short: Price breaks below 20-day low + close < 1w EMA + volume confirmation
# - Exit: Price crosses 10-day EMA (adaptive stop) or opposite Donchian break
# - Position sizing: 0.25 (discrete level to balance return and drawdown)
# - Works in bull/bear: Donchian captures trends, 1w EMA filters counter-trend moves, volume confirms strength
# - Target: 40-80 total trades over 4 years (10-20/year) for 1d timeframe

name = "1d_1w_donchian_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need enough data for indicators
        return np.zeros(n)
    
    # Pre-compute 1d data
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
    # Pre-compute 1w data for EMA trend filter
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian channels (20-period) - based on previous bars to avoid look-ahead
    donchian_high_20 = np.full(len(close_1d), np.nan)
    donchian_low_20 = np.full(len(close_1d), np.nan)
    
    for i in range(20, len(close_1d)):
        donchian_high_20[i] = np.max(high_1d[i-20:i])  # Previous 20 bars, not including current
        donchian_low_20[i] = np.min(low_1d[i-20:i])
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        volume_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Calculate 1w EMA(21) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_21_1w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1d EMA(10) for exit signal
    close_1d_series = pd.Series(close_1d)
    ema_10_1d = close_1d_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Align all HTF indicators to 1d timeframe
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)  # Using 1w alignment for consistency
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1d)  # Using 1w for alignment consistency
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start from 20 to have enough data for Donchian
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_confirm = volume_1d[i] > 1.5 * volume_ma_20_1d[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above 20-day high + close > 1w EMA + volume confirmation
            if close_1d[i] > donchian_high_20_aligned[i] and close_1d[i] > ema_21_1w_aligned[i] and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below 20-day low + close < 1w EMA + volume confirmation
            elif close_1d[i] < donchian_low_20_aligned[i] and close_1d[i] < ema_21_1w_aligned[i] and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price crosses 10-day EMA (adaptive stop) OR opposite Donchian break
            if position == 1:  # Long position
                if close_1d[i] < ema_10_1d[i] or close_1d[i] < donchian_low_20_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_1d[i] > ema_10_1d[i] or close_1d[i] > donchian_high_20_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals