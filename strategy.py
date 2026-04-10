#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Donchian upper band (20-period high) AND 1d close > 1d SMA(50) (bullish trend) AND 1d volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian lower band (20-period low) AND 1d close < 1d SMA(50) (bearish trend) AND 1d volume > 1.5x 20-bar avg
# - Exit when price crosses 12-period SMA on 12h timeframe (trend reversal signal)
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Donchian channels provide clear breakout levels based on price extremes
# - 1d SMA(50) filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# - Volume confirmation avoids low-liquidity false signals
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breakouts in trends, mean reversion in ranges

name = "12h_1d_donchian_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d trend filter: close vs SMA(50)
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    ema_bullish_1d = close_1d > sma_50_1d
    ema_bearish_1d = close_1d < sma_50_1d
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    
    # Pre-compute 12h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 12h exit signal: 12-period SMA
    close = prices['close'].values
    sma_12 = pd.Series(close).rolling(window=12, min_periods=12).mean().values
    
    # Align HTF indicators to 12h timeframe
    ema_bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_bullish_1d)
    ema_bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_bearish_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_1d_aligned[i]) or np.isnan(ema_bearish_1d_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or np.isnan(sma_12[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian upper AND 1d bullish trend AND volume spike
            if (prices['close'].iloc[i] > donchian_upper[i] and 
                ema_bullish_1d_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian lower AND 1d bearish trend AND volume spike
            elif (prices['close'].iloc[i] < donchian_lower[i] and 
                  ema_bearish_1d_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit on trend reversal
            # Exit when price crosses 12-period SMA (trend reversal)
            exit_long = position == 1 and prices['close'].iloc[i] < sma_12[i]
            exit_short = position == -1 and prices['close'].iloc[i] > sma_12[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals