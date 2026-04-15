#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Breakout with Volume Confirmation and Daily Trend Filter
# Long when price breaks above 10-period Donchian upper band + volume spike + daily EMA50 uptrend
# Short when price breaks below 10-period Donchian lower band + volume spike + daily EMA50 downtrend
# Uses volatility-based position sizing (ATR-based) to adapt to market conditions
# Works in bull (breakouts with momentum) and bear (breakdowns with follow-through)
# Target: 15-25 trades/year to minimize fee drag while capturing significant moves

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 12-period Donchian channels (using 12h data for calculation, but applied to 12h timeframe)
    # Since we're on 12h timeframe, we can calculate directly
    donchian_period = 12
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # ATR for volatility-based sizing and stop loss
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: current > 1.8x median of last 24 bars (2 days worth)
    vol_median = pd.Series(volume).rolling(window=24, min_periods=1).median()
    vol_threshold = 1.8 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(donchian_period, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Price breaks above Donchian high + volume spike + daily uptrend
        if (close[i] > donchian_high[i-1] and  # Break above previous period's high
            volume[i] > vol_threshold[i] and
            close[i] > ema_1d_aligned[i]):      # Above daily EMA50 (uptrend)
            # Size based on volatility: higher ATR = smaller position
            vol_factor = min(2.0, max(0.5, 1.5 * atr[i] / close[i]))  # Normalize ATR
            base_size = 0.25
            size = base_size / vol_factor
            size = min(0.35, max(0.15, size))  # Clamp to reasonable range
            signals[i] = size
        
        # Short: Price breaks below Donchian low + volume spike + daily downtrend
        elif (close[i] < donchian_low[i-1] and   # Break below previous period's low
              volume[i] > vol_threshold[i] and
              close[i] < ema_1d_aligned[i]):     # Below daily EMA50 (downtrend)
            vol_factor = min(2.0, max(0.5, 1.5 * atr[i] / close[i]))
            base_size = 0.25
            size = base_size / vol_factor
            size = min(0.35, max(0.15, size))
            signals[i] = -size
        
        # Exit: Price returns to mid-channel or trend fails
        elif i > 0:
            mid_channel = (donchian_high[i] + donchian_low[i]) / 2
            if signals[i-1] > 0 and (close[i] < mid_channel or close[i] < ema_1d_aligned[i]):
                signals[i] = 0.0
            elif signals[i-1] < 0 and (close[i] > mid_channel or close[i] > ema_1d_aligned[i]):
                signals[i] = 0.0
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0