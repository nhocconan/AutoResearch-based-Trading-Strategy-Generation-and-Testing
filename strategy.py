#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# - Long when price breaks above 20-day high AND 1w EMA50 rising AND volume > 2.0x 20-bar avg
# - Short when price breaks below 20-day low AND 1w EMA50 falling AND volume > 2.0x 20-bar avg
# - Exit with ATR(14) trailing stop: long exits when price < highest_high_since_entry - 2.5*ATR, short exits when price > lowest_low_since_entry + 2.5*ATR
# - Uses 1w EMA50 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years)
# - Donchian breakouts capture strong trends; volume filter reduces false breakouts; ATR stop manages risk

name = "1d_1w_donchian_breakout_volume_trend_v1"
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
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    # Pre-compute ATR(14) for trailing stop
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(atr[i])):
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
                highest_high_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short when price breaks below 20-day low AND 1w downtrend with volume spike
            elif (prices['close'].iloc[i] < low_20[i] and 
                  prices['close'].iloc[i] < ema50_1w_aligned[i] and  # price below 1w EMA50
                  vol_spike.iloc[i]):
                position = -1
                lowest_low_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for ATR trailing stop exit
            # Update highest/lowest since entry
            if position == 1:
                highest_high_since_entry = max(highest_high_since_entry, prices['high'].iloc[i])
                # Long exit: price < highest_high_since_entry - 2.5*ATR
                if prices['close'].iloc[i] < highest_high_since_entry - 2.5 * atr[i]:
                    position = 0
                    highest_high_since_entry = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_low_since_entry = min(lowest_low_since_entry, prices['low'].iloc[i])
                # Short exit: price > lowest_low_since_entry + 2.5*ATR
                if prices['close'].iloc[i] > lowest_low_since_entry + 2.5 * atr[i]:
                    position = 0
                    lowest_low_since_entry = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals