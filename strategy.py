#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX(14)>25 trend filter and volume confirmation (>1.5x 20-period EMA)
# Donchian breakouts capture momentum shifts; 1d ADX ensures we only trade in strong trending regimes (works in bull/bear)
# Volume confirmation reduces false breakouts; discrete sizing (0.25) controls fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for optimal risk-adjusted returns

name = "6h_Donchian20_Breakout_1dADX_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: >1.5x 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    # Donchian channels (20-period) on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = pd.Series(df_1d['high']) - pd.Series(df_1d['low'])
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    strong_trend = adx_values > 25
    
    # Align 1d ADX to 6h timeframe (wait for completed 1d bar)
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_confirmation[i]) or np.isnan(strong_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above Donchian upper band with volume confirmation and strong trend
            if close[i] > highest_high[i] and volume_confirmation[i] and strong_trend_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower band with volume confirmation and strong trend
            elif close[i] < lowest_low[i] and volume_confirmation[i] and strong_trend_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below Donchian lower band (reversal to downside)
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above Donchian upper band (reversal to upside)
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals