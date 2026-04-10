#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high AND 1w close > 1w EMA50 AND volume > 1.5x avg
# - Short when price breaks below Donchian(20) low AND 1w close < 1w EMA50 AND volume > 1.5x avg
# - Exit when price crosses Donchian(20) midpoint or 1w trend reverses
# - Uses discrete position sizing (0.25) to control drawdown
# - Targets ~10-20 trades/year (40-80 total over 4 years) to avoid fee drag
# - Donchian breakouts capture strong momentum moves
# - 1w EMA50 filter ensures alignment with higher timeframe trend
# - Volume confirmation prevents false breakouts
# - Works in both bull (strong breakouts) and bear (strong breakdowns) markets

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
    
    # Pre-compute Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high: rolling max of high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of high and low channels
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian high AND 1w uptrend AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema50_1w_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below Donchian low AND 1w downtrend AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price crosses Donchian midpoint (trend weakening)
            # 2. 1w trend reverses (for long: price < 1w EMA50; for short: price > 1w EMA50)
            if position == 1:
                if close[i] < donchian_mid[i] or close[i] < ema50_1w_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if close[i] > donchian_mid[i] or close[i] > ema50_1w_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals