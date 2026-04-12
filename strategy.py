# #!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_keltner_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ATR and weekly high/low context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly ATR (14-period) for volatility context
    tr_w = np.maximum(df_1w['high'].values - df_1w['low'].values,
                      np.maximum(np.abs(df_1w['high'].values - np.concatenate([[np.nan], df_1w['close'].values[:-1]])),
                                 np.abs(df_1w['low'].values - np.concatenate([[np.nan], df_1w['close'].values[:-1]]))))
    atr_14_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    
    # Get daily data for Keltner channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR (10-period) for Keltner channels
    tr_d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                      np.maximum(np.abs(df_1d['high'].values - np.concatenate([[np.nan], df_1d['close'].values[:-1]])),
                                 np.abs(df_1d['low'].values - np.concatenate([[np.nan], df_1d['close'].values[:-1]]))))
    atr_10_d = pd.Series(tr_d).rolling(window=10, min_periods=10).mean().values
    
    # Calculate EMA(20) of daily close for Keltner middle
    ema_20_d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Keltner bands: EMA(20) ± 2 * ATR(10)
    upper_keltner = ema_20_d + 2 * atr_10_d
    lower_keltner = ema_20_d - 2 * atr_10_d
    
    # Align all weekly and daily indicators to 6h timeframe
    atr_14_w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_w)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # Volume filter: 20-period average on 6h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if not ready
        if (np.isnan(atr_14_w_aligned[i]) or np.isnan(upper_keltner_aligned[i]) or
            np.isnan(lower_keltner_aligned[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price breaks above upper Keltner with volume confirmation and low weekly volatility
        # Only trade when weekly ATR is below median (low volatility environment)
        weekly_vol_ok = atr_14_w_aligned[i] < np.nanmedian(atr_14_w_aligned[:i+1])
        long_signal = close[i] > upper_keltner_aligned[i] and volume_ok[i] and weekly_vol_ok
        
        # Short: price breaks below lower Keltner with volume confirmation and low weekly volatility
        short_signal = close[i] < lower_keltner_aligned[i] and volume_ok[i] and weekly_vol_ok
        
        # Exit when price returns to Keltner middle (mean reversion)
        exit_long = close[i] < ema_20_d[np.searchsorted(df_1d.index, prices.index[i], side='right') - 1] if i >= 20 else False
        exit_short = close[i] > ema_20_d[np.searchsorted(df_1d.index, prices.index[i], side='right') - 1] if i >= 20 else False
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals