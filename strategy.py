#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_volume_trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h OHLC for 4h context
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h ATR for volatility filter
    tr1_12h = high_12h[1:] - low_12h[1:]
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # 12h volume filter: volume > 1.5x 20-period average
    vol_ma_20_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # 12h ADX for trend strength
    plus_dm_12h = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm_12h = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    tr_dm_12h = tr_12h[1:]
    plus_di_12h = 100 * pd.Series(plus_dm_12h).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm_12h).rolling(window=14, min_periods=14).mean().values
    minus_di_12h = 100 * pd.Series(minus_dm_12h).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm_12h).rolling(window=14, min_periods=14).mean().values
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = pd.Series(dx_12h).rolling(window=14, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 4h ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_12h_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Volume confirmation (1.5x 12h average)
        vol_ma = vol_ma_20_12h_aligned[i]
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filter: ADX > 25 (strong trend)
        trend_filter = adx_12h_aligned[i] > 25
        
        # Breakout levels: 4h price breaking 12h ATR-based channels
        upper_channel = close_12h[-1] + 0.5 * atr_12h_aligned[i]  # Using last known 12h close
        lower_channel = close_12h[-1] - 0.5 * atr_12h_aligned[i]
        
        # Long conditions: price breaks above upper channel with volume and trend
        long_signal = volume_confirmed and trend_filter and (price_high > upper_channel)
        
        # Short conditions: price breaks below lower channel with volume and trend
        short_signal = volume_confirmed and trend_filter and (price_low < lower_channel)
        
        # Stoploss: 2 * 4h ATR
        if position == 1 and price_close < entry_price - 2.0 * atr_4h[i]:
            position = 0
            signals[i] = 0.0
            continue
        elif position == -1 and price_close > entry_price + 2.0 * atr_4h[i]:
            position = 0
            signals[i] = 0.0
            continue
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and price_close < lower_channel:  # Exit long on reversion to lower channel
            position = 0
            signals[i] = 0.0
        elif position == -1 and price_close > upper_channel:  # Exit short on reversion to upper channel
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h breakout strategy using 12h ATR-based channels with volume confirmation and ADX filter.
# Enters long when 4h price breaks above 12h close + 0.5*ATR(12h) with volume >1.5x 12h volume average and ADX(12h)>25.
# Enters short when price breaks below 12h close - 0.5*ATR(12h) with same conditions.
# Exits when price reverts to the opposite channel or stoploss is hit.
# Uses 12h timeframe for trend and volatility context to avoid 4h noise.
# Volume and ADX filters reduce false breakouts and overtrading.
# Target: 20-40 trades per year to minimize fee drag while capturing strong 12h trends.
# Designed to work in both bull (breakouts continuation) and bear (breakdowns continuation) markets.