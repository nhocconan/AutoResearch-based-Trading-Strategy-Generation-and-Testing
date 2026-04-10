#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high AND weekly close > weekly open (bullish weekly bias) AND volume > 1.5x 20-period volume SMA
# - Short when price breaks below Donchian(20) low AND weekly close < weekly open (bearish weekly bias) AND volume > 1.5x 20-period volume SMA
# - Exit: opposite Donchian breakout or volume drops below average
# - Uses 1d for Donchian and volume, 1w for trend bias
# - Weekly trend filter ensures we only trade in the direction of the higher timeframe momentum
# - Volume confirmation ensures breakouts have conviction
# - Target: 10-25 trades/year to minimize fee drag while capturing meaningful moves

name = "1d_1w_donchian_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for weekly trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return signals
    
    # Calculate weekly trend: bullish if weekly close > weekly open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # True for bullish weekly candle
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Pre-compute Donchian channels for 1d data (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume SMA for 1d data (20-period)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after 20-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_sma_20[i]) or np.isnan(weekly_bullish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_high[i-1]  # Break above prior period's high
        breakout_short = close[i] < donchian_low[i-1]  # Break below prior period's low
        
        # Weekly trend filter
        weekly_trend_bullish = weekly_bullish_aligned[i] > 0.5  # Convert to boolean
        weekly_trend_bearish = weekly_bullish_aligned[i] <= 0.5  # Convert to boolean
        
        # Exit conditions: opposite breakout or volume drops below average
        exit_long = close[i] < donchian_low[i-1] or volume[i] < volume_sma_20[i]
        exit_short = close[i] > donchian_high[i-1] or volume[i] < volume_sma_20[i]
        
        # Trading logic
        if vol_confirm:
            # Long: Donchian breakout above with bullish weekly trend
            if breakout_long and weekly_trend_bullish:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: Donchian breakout below with bearish weekly trend
            elif breakout_short and weekly_trend_bearish:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Check for exits
                if position == 1 and exit_long:
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No volume confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals