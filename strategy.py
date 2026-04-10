#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and volume confirmation
# - Long when price breaks above 20-day high AND 1w EMA50 rising AND volume > 1.5x 20-bar avg
# - Short when price breaks below 20-day low AND 1w EMA50 falling AND volume > 1.5x 20-bar avg
# - Exit when price crosses 10-day EMA in opposite direction (trend reversal signal)
# - Uses 1w EMA50 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Donchian breakouts work well in trending markets; trend filter adds robustness in ranging/volatile markets

name = "1d_20d_donchian_breakout_1w_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels from daily data (20-period)
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 10-day EMA for exit signal
    close_prices = prices['close'].values
    ema10 = pd.Series(close_prices).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema10[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above 20-day high AND 1w uptrend with volume spike
            if (prices['close'].iloc[i] > high_20[i] and 
                prices['close'].iloc[i] > ema50_1w_aligned[i] and  # price above 1w EMA50
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below 20-day low AND 1w downtrend with volume spike
            elif (prices['close'].iloc[i] < low_20[i] and 
                  prices['close'].iloc[i] < ema50_1w_aligned[i] and  # price below 1w EMA50
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit on trend reversal
            # Exit when price crosses 10-day EMA in opposite direction
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] < ema10[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] > ema10[i]:
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