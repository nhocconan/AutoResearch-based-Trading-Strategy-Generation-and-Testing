#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and volume confirmation
# - Long when price breaks above 20-bar Donchian high AND 4h EMA50 rising AND volume > 2.0x 20-bar avg
# - Short when price breaks below 20-bar Donchian low AND 4h EMA50 falling AND volume > 2.0x 20-bar avg
# - Exit when price returns to 20-bar Donchian midpoint (mean reversion to equilibrium)
# - Uses 4h EMA50 for trend filter to avoid counter-trend trades
# - Session filter: 08-20 UTC to reduce noise trades
# - Discrete position sizing (0.20) to minimize fee churn
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Donchian channels work well in trending markets; trend filter adds robustness in bear markets

name = "1h_4h_donchian_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period)
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # Pre-compute 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(volume_20_avg[i]) or not in_session.iloc[i]):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND 4h uptrend with volume spike
            if (prices['close'].iloc[i] > high_20[i] and 
                prices['close'].iloc[i] > ema50_4h_aligned[i] and  # price above 4h EMA50
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.20
            # Short when price breaks below Donchian low AND 4h downtrend with volume spike
            elif (prices['close'].iloc[i] < low_20[i] and 
                  prices['close'].iloc[i] < ema50_4h_aligned[i] and  # price below 4h EMA50
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to midpoint (mean reversion)
            # Exit when price returns to Donchian midpoint
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
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals