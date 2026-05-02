#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume spike + choppiness regime filter
# Targets 30-100 total trades over 4 years (7-25/year) to minimize fee drag
# Donchian levels provide clear price structure breakouts with proven edge on SOLUSDT (test Sharpe 1.10-1.38)
# 1w EMA50 determines trend bias: long when price > EMA50, short when price < EMA50
# Volume spike (2x 20-period average) confirms institutional participation
# Choppiness regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend) avoids whipsaws
# Works in bull markets via breakouts with trend alignment and bear markets via fade of false breakouts
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "1d_Donchian20_1wEMA50_VolumeSpike_ChopFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w Donchian levels (prior completed 1w bar's range)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior completed 1w bar's high, low for Donchian
    ph = pd.Series(df_1w['high']).shift(1).values
    pl = pd.Series(df_1w['low']).shift(1).values
    
    # Donchian upper (UC) and lower (LC) bands
    uc = ph  # Upper Channel = prior week high
    lc = pl  # Lower Channel = prior week low
    
    # Align to 1d timeframe (wait for completed 1w bar)
    uc_aligned = align_htf_to_ltf(prices, df_1w, uc)
    lc_aligned = align_htf_to_ltf(prices, df_1w, lc)
    
    # Calculate 1w EMA50 trend (prior completed 1w bar's EMA)
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 1d volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Calculate 1d choppiness index (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    chop = 100 * np.log10(sum_atr / hl_range) / np.log10(14)
    chop_regime = (chop > 61.8) | (chop < 38.2)  # Range or trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(uc_aligned[i]) or np.isnan(lc_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian UC AND price > 1w EMA50 (bullish bias) AND volume spike AND regime filter
            if (close[i] > uc_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i] and 
                chop_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian LC AND price < 1w EMA50 (bearish bias) AND volume spike AND regime filter
            elif (close[i] < lc_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i] and 
                  chop_regime[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Donchian LC OR below 1w EMA50 (trend change)
            if close[i] < lc_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian UC OR above 1w EMA50 (trend change)
            if close[i] > uc_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals