# Hypothesis: Use 12h timeframe with 1d/1w HTF filters (Donchian breakout + volume + EMA200 trend + ATR stop) to capture multi-day trends while minimizing trades. Works in bull/bear via trend filter and volatility-based sizing.
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Donchian channels (20-period) - using previous bar's high/low
    high_series_1d = pd.Series(high_1d)
    low_series_1d = pd.Series(low_1d)
    upper_1d = high_series_1d.rolling(window=20, min_periods=20).max().shift(1).values
    lower_1d = low_series_1d.rolling(window=20, min_periods=20).min().shift(1).values
    
    # 1d average volume (20-period) - previous bar
    vol_series_1d = pd.Series(volume_1d)
    avg_vol_1d = vol_series_1d.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # 1d EMA200 trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # 1d ATR (14-period) for stop-loss
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    tr_1d[0] = high_low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().shift(1).values
    
    # Align HTF indicators to 12h timeframe (waiting for close)
    upper = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower = align_htf_to_ltf(prices, df_1d, lower_1d)
    avg_vol = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    ema_200 = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    atr = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    start = max(20, 200, 14)
    for i in range(start, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(avg_vol[i]) or np.isnan(ema_200[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: breakout above upper band + volume confirmation + price above EMA200
            if (price > upper[i] and vol > 2.0 * avg_vol[i] and price > ema_200[i]):
                position = 1
                signals[i] = position_size
            # Short: breakout below lower band + volume confirmation + price below EMA200
            elif (price < lower[i] and vol > 2.0 * avg_vol[i] and price < ema_200[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower band OR below EMA200 OR stop-loss hit
            if (price < lower[i] or price < ema_200[i] or 
                price < entry_price_long - 2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper band OR above EMA200 OR stop-loss hit
            if (price > upper[i] or price > ema_200[i] or 
                price > entry_price_short + 2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        
        # Track entry price for stop-loss calculation
        if position != 0 and signals[i] != 0 and (i == start or signals[i-1] == 0):
            if position == 1:
                entry_price_long = close[i]
            else:
                entry_price_short = close[i]
    
    return signals

name = "12h_1d_Donchian_Volume_EMA200Trend_ATR"
timeframe = "12h"
leverage = 1.0