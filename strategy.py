#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter + volume confirmation
# - Long when price breaks above 20-day high with volume > 1.5x 20-day avg AND 1w close > 1w EMA50
# - Short when price breaks below 20-day low with volume > 1.5x 20-day avg AND 1w close < 1w EMA50
# - Exit when price retreats to midpoint of 20-day channel OR ATR-based stoploss hit
# - Uses 1w trend filter to avoid counter-trend trades in bear markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years)

name = "1d_1w_donchian_breakout_volume_trend_v1"
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
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute ATR(14) for stoploss
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    tr = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0  # track entry price for stoploss
    
    # Pre-compute aligned 1w data
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    h_1w_aligned = align_htf_to_ltf(prices, df_1w, h_1w)
    l_1w_aligned = align_htf_to_ltf(prices, df_1w, l_1w)
    c_1w_aligned = align_htf_to_ltf(prices, df_1w, c_1w)
    
    # Pre-compute 1w EMA(50) for trend filter
    ema50_1w = pd.Series(c_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(atr[i]) or np.isnan(h_1w_aligned[i]) or np.isnan(l_1w_aligned[i]) or 
            np.isnan(c_1w_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get Donchian channel from 20-day lookback (completed bars only)
        lookback_start = i - 20
        if lookback_start >= 0:
            period_high = np.max(prices['high'].iloc[lookback_start:i].values)
            period_low = np.min(prices['low'].iloc[lookback_start:i].values)
            
            if position == 0:  # Flat - look for new breakout entries
                # Long breakout: price > 20-day high with volume spike AND 1w uptrend
                if (prices['high'].iloc[i] > period_high and 
                    vol_spike.iloc[i] and 
                    prices['close'].iloc[i] > ema50_1w_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Short breakdown: price < 20-day low with volume spike AND 1w downtrend
                elif (prices['low'].iloc[i] < period_low and 
                      vol_spike.iloc[i] and 
                      prices['close'].iloc[i] < ema50_1w_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:  # Have position - look for exit
                # Exit conditions:
                # 1. Price retreats to midpoint of 20-day channel
                # 2. ATR-based stoploss hit
                midpoint = (period_high + period_low) / 2
                exit_signal = False
                if position == 1:  # Long position
                    if (prices['close'].iloc[i] < midpoint or
                        prices['close'].iloc[i] < entry_price - 2.5 * atr[i]):
                        exit_signal = True
                elif position == -1:  # Short position
                    if (prices['close'].iloc[i] > midpoint or
                        prices['close'].iloc[i] > entry_price + 2.5 * atr[i]):
                        exit_signal = True
                
                if exit_signal:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    if position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
        else:
            # Not enough data yet, hold flat
            signals[i] = 0.0
    
    return signals