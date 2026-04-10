#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and chop regime filter
# - Primary: 12h timeframe for lower frequency and reduced fee drag
# - HTF: 1d for volume confirmation (above average) and chop regime filter (CHOP > 61.8 = range)
# - Long: Price breaks above 20-period Donchian high + 1d volume > 1.2x 20-period MA + CHOP > 61.8
# - Short: Price breaks below 20-period Donchian low + 1d volume > 1.2x 20-period MA + CHOP > 61.8
# - Exit: Price reverts to 10-period EMA (mean reversion in ranging markets)
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) - within 12h sweet spot
# - Works in bull/bear: Donchian breakouts capture trends, chop filter ensures we only trade in ranging markets where mean reversion works

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Donchian Channel (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h 10-period EMA for exit signal
    close_s = pd.Series(close_12h)
    ema_10 = close_s.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(atr1..atr14) / (atr14 * log2(N+1))) / log2(N)
    # where N = 14, simplified: CHOP = 100 * log10(sum(atr) / (atr14 * log2(15))) / log2(14)
    high_low = pd.Series(high_1d) - pd.Series(low_1d)
    high_close = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    low_close = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Sum of ATR over 14 periods
    sum_atr = atr.rolling(window=14, min_periods=14).sum()
    # Choppiness Index formula
    chop = 100 * np.log10(sum_atr / (atr * np.log2(15))) / np.log2(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_10[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1d volume confirmation: current volume > 1.2x 20-period MA
        volume_confirm = volume_1d[i] > 1.2 * volume_ma_20_1d_aligned[i]
        
        # Chop regime filter: CHOP > 61.8 = ranging market (good for mean reversion)
        chop_regime = chop_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian high + volume confirm + chop regime
            if (close_12h[i] > donchian_high[i] and volume_confirm and chop_regime):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low + volume confirm + chop regime
            elif (close_12h[i] < donchian_low[i] and volume_confirm and chop_regime):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price reverts to 10-period EMA (mean reversion)
            if position == 1:  # Long position
                if close_12h[i] < ema_10[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_12h[i] > ema_10[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals