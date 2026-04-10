#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high + 1w uptrend (close > EMA50) + volume spike
# - Short when price breaks below Donchian(20) low + 1w downtrend (close < EMA50) + volume spike
# - Uses 1d timeframe targeting 7-25 trades/year (30-100 total over 4 years) to minimize fee drag
# - 1w EMA50 filter ensures trading with higher timeframe trend direction
# - 1d volume > 2.0x 20-period average confirms breakout strength
# - Discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14) or opposite Donchian breakout

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
    
    # 1w volume confirmation: > 2.0x 20-period average
    avg_volume_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = volume_1w > (2.0 * avg_volume_20_1w)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # 1w ATR(14) for stoploss
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
    
    # 1d Donchian(20) channels
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Upper channel: highest high of last 20 periods
    upper_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 periods
    lower_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1d ATR(14) for stoploss
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_14_1d = np.zeros_like(tr_1d)
    atr_14_1d[14-1] = np.mean(tr_1d[:14])
    for i in range(14, len(tr_1d)):
        atr_14_1d[i] = (atr_14_1d[i-1] * (14-1) + tr_1d[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike_1w_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(atr_14_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or opposite Donchian breakout (below lower channel)
            if (prices['close'].iloc[i] < entry_price - 2.0 * entry_atr or 
                prices['close'].iloc[i] < lower_channel[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or opposite Donchian breakout (above upper channel)
            if (prices['close'].iloc[i] > entry_price + 2.0 * entry_atr or 
                prices['close'].iloc[i] > upper_channel[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike_1w_aligned[i]:
                # Long signal: break above upper channel in 1w uptrend
                if prices['close'].iloc[i] > upper_channel[i] and prices['close'].iloc[i] > ema_50_1w_aligned[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d[i]
                    signals[i] = 0.25
                # Short signal: break below lower channel in 1w downtrend
                elif prices['close'].iloc[i] < lower_channel[i] and prices['close'].iloc[i] < ema_50_1w_aligned[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d[i]
                    signals[i] = -0.25
    
    return signals