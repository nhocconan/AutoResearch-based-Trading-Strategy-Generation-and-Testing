#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(50) trend filter and 1w ATR(14) volume confirmation
# - Primary: 1d price breaks above Donchian(20) high (long) or below Donchian(20) low (short)
# - Trend filter: 1w EMA(50) direction (price above/below EMA for long/short bias) - avoids counter-trend trades
# - Volume filter: 1w ATR(14) > 1.3x its 50-period MA to confirm institutional participation
# - Exit: Close crosses back inside Donchian channel (mean reversion exit) - captures swings in both bull/bear
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# - Works in bull/bear: Donchian breakouts capture trends, EMA filter avoids whipsaws, ATR volume avoids fakeouts

name = "1d_1w_donchian_ema_atr_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute HTF data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on 1d
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w ATR(14) for volume confirmation
    high_low_1w = high_1w - low_1w
    high_close_1w = np.abs(high_1w - np.roll(close_1w, 1))
    low_close_1w = np.abs(low_1w - np.roll(close_1w, 1))
    high_close_1w[0] = high_low_1w[0]
    low_close_1w[0] = high_low_1w[0]
    tr_1w = np.maximum(high_low_1w, np.maximum(high_close_1w, low_close_1w))
    atr_14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_ma_50_1w = pd.Series(atr_14_1w).rolling(window=50, min_periods=50).mean().values
    atr_ma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_ma_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w ATR(14) > 1.3x its 50-period MA
        atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
        volume_confirm = atr_14_1w_aligned[i] > 1.3 * atr_ma_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: breakout above upper Donchian band + price above 1w EMA(50) + volume confirmation
            if (close[i] > highest_20[i] and close[i] > ema_50_1w_aligned[i] and volume_confirm):
                position = 1
                signals[i] = 0.25
            # Short entry: breakout below lower Donchian band + price below 1w EMA(50) + volume confirmation
            elif (close[i] < lowest_20[i] and close[i] < ema_50_1w_aligned[i] and volume_confirm):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Close crosses back inside Donchian channel (mean reversion exit)
            if position == 1:  # Long position
                if close[i] < lowest_20[i]:  # Exit when price breaks below lower band
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > highest_20[i]:  # Exit when price breaks above upper band
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals