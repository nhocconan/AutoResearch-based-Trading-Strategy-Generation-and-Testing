#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d trend filter (EMA50>EMA200) and volume confirmation
# - Long when price breaks above Donchian upper channel (20) AND 1d EMA(50) > EMA(200) AND volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian lower channel (20) AND 1d EMA(50) < EMA(200) AND volume > 1.5x 20-bar avg
# - Exit when price touches the opposite Donchian channel (mean reversion)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Donchian channels provide clear structure for breakouts in both bull and bear markets
# - 1d EMA filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# - Volume confirmation filters out low-liquidity false breakouts
# - Target: 15-40 trades/year on 4h timeframe (60-160 total over 4 years)

name = "4h_1d_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA trend filter: EMA(50) vs EMA(200)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish_1d = ema_50_1d > ema_200_1d
    ema_bearish_1d = ema_50_1d < ema_200_1d
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    
    # Pre-compute 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper and lower channels
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 4h timeframe
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
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
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
            # Long when price breaks above Donchian upper channel AND 1d bullish trend AND volume spike
            if (close[i] > donchian_upper[i] and 
                ema_bullish_1d_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian lower channel AND 1d bearish trend AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  ema_bearish_1d_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to opposite channel (mean reversion)
            # Exit when price touches opposite Donchian channel
            exit_long = position == 1 and close[i] <= donchian_lower[i]
            exit_short = position == -1 and close[i] >= donchian_upper[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals