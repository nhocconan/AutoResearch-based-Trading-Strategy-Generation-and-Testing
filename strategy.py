#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w ADX regime + volume confirmation
# In trending markets (ADX>=25), we trade breakouts in trend direction: long on upper breakout in uptrend, short on lower breakout in downtrend.
# In ranging markets (ADX<25), we fade extremes: short near upper band, long near lower band.
# Volume confirmation (>1.5x 20-period EMA) reduces false breakouts. Designed for 1d timeframe targeting 30-100 total trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "1d_Donchian20_1wADX_Regime_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ADX (14-period)
    plus_dm = pd.Series(df_1w['high']).diff()
    minus_dm = pd.Series(df_1w['low']).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = pd.Series(df_1w['high']).sub(df_1w['low'])
    tr2 = pd.Series(df_1w['high']).sub(df_1w['close'].shift(1)).abs()
    tr3 = pd.Series(df_1w['low']).sub(df_1w['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Calculate 1w Donchian channels (20-period)
    highest_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Align 1w indicators to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx.values)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Volume confirmation: 20-period EMA of volume on 1d timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Determine regime: ranging (ADX<25) or trending (ADX>=25)
            if adx_aligned[i] < 25:
                # Ranging market: fade extremes (mean reversion)
                if close[i] <= donchian_lower_aligned[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= donchian_upper_aligned[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            else:
                # Trending market: trade breakouts in trend direction
                # Trend direction: +DI > -DI indicates uptrend
                plus_di_1w = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
                minus_di_1w = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
                plus_di_aligned = align_htf_to_ltf(prices, df_1w, plus_di_1w.values)
                minus_di_aligned = align_htf_to_ltf(prices, df_1w, minus_di_1w.values)
                
                # Long: upper breakout in uptrend (+DI > -DI)
                if (close[i] > donchian_upper_aligned[i] and 
                    volume_confirm and 
                    plus_di_aligned[i] > minus_di_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: lower breakout in downtrend (-DI > +DI)
                elif (close[i] < donchian_lower_aligned[i] and 
                      volume_confirm and 
                      minus_di_aligned[i] > plus_di_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price retouches midpoint OR ADX weakening (<20) OR volume drops
            if (close[i] <= donchian_mid_aligned[i] or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches midpoint OR ADX weakening (<20) OR volume drops
            if (close[i] >= donchian_mid_aligned[i] or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals