#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation
# - Long when price breaks above Donchian(20) high AND price > 1d EMA50 (bullish trend) AND 1d volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian(20) low AND price < 1d EMA50 (bearish trend) AND 1d volume > 1.5x 20-bar avg
# - Exit when price crosses 1d EMA21 (mean reversion to intermediate trend)
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Donchian captures structural breaks; 1d EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breaks in trending markets, EMA exit prevents whipsaw

name = "12h_1d_donchian_ema_volume_trend_v1"
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
    
    # Pre-compute 1d EMA trend filter: EMA21 and EMA50
    close_1d = df_1d['close'].values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    
    # Align HTF indicators to 12h timeframe
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute Donchian channels on 12h data: 20-period high/low
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Breakout conditions
    breakout_long = prices['close'].values > donchian_high  # Close above upper band
    breakout_short = prices['close'].values < donchian_low   # Close below lower band
    
    # EMA trend conditions
    price_above_ema21 = prices['close'].values > ema_21_1d_aligned
    price_below_ema21 = prices['close'].values < ema_21_1d_aligned
    price_above_ema50 = prices['close'].values > ema_50_1d_aligned
    price_below_ema50 = prices['close'].values < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_21_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when breakout above Donchian high AND price > EMA50 (bullish trend) AND volume spike
            if (breakout_long[i] and 
                price_above_ema50[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when breakout below Donchian low AND price < EMA50 (bearish trend) AND volume spike
            elif (breakout_short[i] and 
                  price_below_ema50[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when price crosses EMA21
            # Exit when price crosses 1d EMA21 (mean reversion to intermediate trend)
            if position == 1 and price_below_ema21[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and price_above_ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals