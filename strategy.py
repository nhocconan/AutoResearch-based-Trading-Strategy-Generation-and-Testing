#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_cloud_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return signals
    
    # Ichimoku components on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted -22 periods (not used for signals)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 6m ATR for volatility filter
    tr1 = pd.Series(high).rolling(window=1).max() - pd.Series(low).rolling(window=1).min()
    tr2 = abs(pd.Series(high).rolling(window=1).max() - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low).rolling(window=1).min() - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        tenkan_val = tenkan_6h[i]
        kijun_val = kijun_6h[i]
        senkou_a_val = senkou_a_6h[i]
        senkou_b_val = senkou_b_6h[i]
        
        # Cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        # Ichimoku signals
        tk_cross_bull = tenkan_val > kijun_val
        tk_cross_bear = tenkan_val < kijun_val
        
        # Price relative to cloud
        price_above_cloud = price_close > upper_cloud
        price_below_cloud = price_close < lower_cloud
        price_in_cloud = (price_close >= lower_cloud) and (price_close <= upper_cloud)
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr[i] > 0.01 * price_close  # ATR > 1% of price
        
        # Trading logic
        long_signal = False
        short_signal = False
        
        # Long: TK bullish cross + price above cloud + volatility filter
        if tk_cross_bull and price_above_cloud and vol_filter:
            long_signal = True
        
        # Short: TK bearish cross + price below cloud + volatility filter
        if tk_cross_bear and price_below_cloud and vol_filter:
            short_signal = True
        
        # Exit: TK cross in opposite direction or price enters cloud
        exit_long = (tk_cross_bear) or (price_in_cloud)
        exit_short = (tk_cross_bull) or (price_in_cloud)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6h Ichimoku cloud strategy with TK cross signals and cloud filter from 1d.
# Enters long when Tenkan crosses above Kijun AND price is above the cloud (bullish trend).
# Enters short when Tenkan crosses below Kijun AND price is below the cloud (bearish trend).
# Uses daily Ichimoku for higher timeframe trend context, reducing whipsaws.
# Volatility filter (ATR > 1% of price) avoids choppy markets.
# Works in bull markets via trend-following signals and in bear markets via short signals.
# Cloud acts as dynamic support/resistance, improving entry/exit timing.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.