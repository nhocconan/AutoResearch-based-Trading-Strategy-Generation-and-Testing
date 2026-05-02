#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Donchian(20) from 1d chart captures medium-term structure - breaks often lead to sustained moves
# 1w EMA34 > price for long bias, < price for short bias ensures we trade with the weekly trend
# Volume spike (>1.5 x 20-period EMA) confirms breakout validity with strong participation
# Discrete position sizing (0.25) controls fee drag while allowing meaningful exposure
# Target: 30-100 total trades over 4 years (7-25/year) for optimal risk-adjusted returns
# Works in bull markets by catching breakouts with trend, works in bear by only taking trend-aligned breaks
# Focus on BTC/ETH as primary symbols

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (volume spike > 1.5 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    # 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    # 1w EMA34 calculation
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 1d timeframe (wait for completed 1w bar)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1d data for Donchian(20) calculation
    if len(high) < 20 or len(low) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels from previous 20 periods (excluding current bar)
    # Using rolling window with min_periods to ensure no look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Donchian upper: highest high of previous 20 bars
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    # Donchian lower: lowest low of previous 20 bars
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA34
        bullish_trend = close[i] > ema_34_1w_aligned[i]
        bearish_trend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above Donchian high with volume confirmation and bullish trend
            if close[i] > donchian_high[i] and volume_confirmation[i] and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian low with volume confirmation and bearish trend
            elif close[i] < donchian_low[i] and volume_confirmation[i] and bearish_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below Donchian low (reversal to downside)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above Donchian high (reversal to upside)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals