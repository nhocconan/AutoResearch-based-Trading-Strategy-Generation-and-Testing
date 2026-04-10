#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA50>EMA200) and volume confirmation
# - Long when close breaks above Donchian(20) high AND 1d EMA50 > EMA200 AND 4h volume > 1.5x 20-bar avg
# - Short when close breaks below Donchian(20) low AND 1d EMA50 < EMA200 AND 4h volume > 1.5x 20-bar avg
# - Exit when price crosses opposite Donchian channel (long exits on low break, short exits on high break)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Donchian breakouts capture strong momentum moves; 1d EMA filter ensures alignment with daily trend
# - Volume confirmation avoids low-liquidity false breakouts
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Works in both bull and bear markets: trend filter prevents counter-trend trades in bear, breakouts capture moves in bull

name = "4h_1d_donchian_breakout_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA trend filter: EMA(50) vs EMA(200)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish = ema_50 > ema_200
    ema_bearish = ema_50 < ema_200
    
    # Align 1d EMA trend to 4h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_bearish)
    
    # Pre-compute Donchian(20) channels on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND 1d bullish trend AND volume spike
            if (close[i] > donchian_high[i-1] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND 1d bearish trend AND volume spike
            elif (close[i] < donchian_low[i-1] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit on opposite channel break
            # Exit long when price breaks below Donchian low
            # Exit short when price breaks above Donchian high
            exit_long = close[i] < donchian_low[i-1]
            exit_short = close[i] > donchian_high[i-1]
            
            if (position == 1 and exit_long) or (position == -1 and exit_short):
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals