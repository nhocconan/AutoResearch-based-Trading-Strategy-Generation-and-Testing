#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# - Long: price breaks above 20-period 12h Donchian high with 1d EMA(50) uptrend and volume spike
# - Short: price breaks below 20-period 12h Donchian low with 1d EMA(50) downtrend and volume spike
# - Exit: price returns to opposite Donchian band (mean reversion at channel midpoint)
# - Uses 12h primary timeframe with 1d HTF for trend filter
# - Volume confirmation: current volume > 2.0x 20-period 12h volume average
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Works in both bull and bear markets by combining breakout with trend filter

name = "12h_1d_donchian_breakout_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 12h Donchian channels (20-period)
    # Donchian high = max(high, lookback=20)
    # Donchian low = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Donchian levels
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        
        # 1d trend filter
        ema_50 = ema_50_1d_aligned[i]
        trend_up = close_price > ema_50
        trend_down = close_price < ema_50
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above upper band with volume and uptrend
        if close_price > upper_band and vol_confirm and trend_up:
            enter_long = True
        
        # Short breakout: price breaks below lower band with volume and downtrend
        if close_price < lower_band and vol_confirm and trend_down:
            enter_short = True
        
        # Exit conditions: mean reversion at opposite band
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price drops back to lower band
            exit_long = close_price <= lower_band
        elif position == -1:
            # Exit short if price rises back to upper band
            exit_short = close_price >= upper_band
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals