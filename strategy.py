#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Donchian(20) upper/lower bands on daily closes for breakout signals
# - 1w EMA(50) as higher timeframe trend filter: long only in uptrend, short only in downtrend
# - Daily volume > 1.5x 20-period average as confirmation of breakout strength
# - ATR-based stoploss: exit when price moves against position by 2.5x ATR(14)
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 30-100 trades over 4 years (7-25/year) to avoid fee drag
# - Works in bull markets via breakouts with trend, in bear markets via short breakdowns

name = "1d_1w_donchian_volume_trend_atr_v1"
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
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w ATR(14) for stoploss calculation (used for both timeframes)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1w = np.zeros_like(tr)
    atr_14_1w[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1w[i] = (atr_14_1w[i-1] * (14-1) + tr[i]) / 14
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Daily Donchian(20) bands
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian upper/lower bands (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily volume confirmation: > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * avg_volume_20)
    
    # Daily ATR(14) for stoploss
    tr1_d = high - low
    tr2_d = np.abs(high - np.roll(close, 1))
    tr3_d = np.abs(low - np.roll(close, 1))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d[0] = tr1_d[0]
    atr_14_d = np.zeros_like(tr_d)
    atr_14_d[14-1] = np.mean(tr_d[:14])
    for i in range(14, len(tr_d)):
        atr_14_d[i] = (atr_14_d[i-1] * (14-1) + tr_d[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(atr_14_d[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price breaks below Donchian lower
            if (prices['close'].iloc[i] < entry_price - 2.5 * entry_atr or 
                prices['close'].iloc[i] < donchian_lower[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price breaks above Donchian upper
            if (prices['close'].iloc[i] > entry_price + 2.5 * entry_atr or 
                prices['close'].iloc[i] > donchian_upper[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i]:
                # Long signal: price breaks above Donchian upper in 1w uptrend
                if prices['close'].iloc[i] > donchian_upper[i] and prices['close'].iloc[i] > ema_50_1w_aligned[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_d[i]
                    signals[i] = 0.25
                # Short signal: price breaks below Donchian lower in 1w downtrend
                elif prices['close'].iloc[i] < donchian_lower[i] and prices['close'].iloc[i] < ema_50_1w_aligned[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_d[i]
                    signals[i] = -0.25
    
    return signals