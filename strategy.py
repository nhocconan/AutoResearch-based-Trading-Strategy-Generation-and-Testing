#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# - Long when price breaks above Donchian(20) upper band AND 12h EMA50 rising AND volume > 1.8x 30-bar avg
# - Short when price breaks below Donchian(20) lower band AND 12h EMA50 falling AND volume > 1.8x 30-bar avg
# - Exit when price touches Donchian(20) midpoint (mean reversion to equilibrium)
# - Uses 12h EMA50 for trend filter to avoid counter-trend trades in bear markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_12h_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels from primary timeframe (4h)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian(20) channels
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Pre-compute volume confirmation: > 1.8x 30-period average
    volume_30_avg = prices['volume'].rolling(window=30, min_periods=30).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_30_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_30_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian upper AND 12h uptrend with volume spike
            if (prices['close'].iloc[i] > donchian_high[i] and 
                close_4h[i] > ema50_12h_aligned[i] and  # price above 12h EMA50
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian lower AND 12h downtrend with volume spike
            elif (prices['close'].iloc[i] < donchian_low[i] and 
                  close_4h[i] < ema50_12h_aligned[i] and  # price below 12h EMA50
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to midpoint (mean reversion)
            # Exit when price touches Donchian midpoint
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= donchian_mid[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals