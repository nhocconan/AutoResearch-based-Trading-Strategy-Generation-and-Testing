#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 1D CCI Trend + Volume + 1D ATR Volatility Filter
# Uses daily CCI (20) for trend direction (long when CCI > 0, short when CCI < 0),
# confirmed by 4h price closing above/below 20-period EMA and volume > 1.5x 20-period average.
# Filters out low-volatility environments using 1D ATR ratio (current ATR / 20-period ATR < 0.5).
# Designed for 15-25 trades/year to minimize fee drag while capturing sustained trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1D data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1D CCI (20-period)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    sma_tp = typical_price.rolling(window=20, min_periods=20).mean()
    mad = typical_price.rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci_1d = (typical_price - sma_tp) / (0.015 * mad)
    cci_1d = cci_1d.fillna(0).values
    
    # Calculate 1D ATR (14-period) and its 20-period average for volatility filter
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift())
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift())
    tr_1d = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean()
    atr_ma_1d = atr_1d.rolling(window=20, min_periods=20).mean()
    atr_ratio_1d = (atr_1d / atr_ma_1d).fillna(1.0).values  # Avoid division by zero
    
    # Align 1D indicators to 4H timeframe
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Pre-compute 4H indicators
    ema_20 = prices['close'].ewm(span=20, min_periods=20, adjust=False).mean().values
    vol_ma = prices['volume'].rolling(window=20, min_periods=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(cci_1d_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i]) or
            np.isnan(ema_20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Volatility filter: only trade when volatility is elevated (ATR ratio > 0.5)
        volatility_filter = atr_ratio_1d_aligned[i] > 0.5
        
        # Trend direction from 1D CCI
        trend_long = cci_1d_aligned[i] > 0
        trend_short = cci_1d_aligned[i] < 0
        
        # Price position relative to 4H EMA
        price_above_ema = price > ema_20[i]
        price_below_ema = price < ema_20[i]
        
        if position == 0:
            # Enter long: daily uptrend + price above 4H EMA + volume + volatility
            if trend_long and price_above_ema and volume_confirm and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: daily downtrend + price below 4H EMA + volume + volatility
            elif trend_short and price_below_ema and volume_confirm and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when daily trend turns down OR price breaks below 4H EMA
                if not trend_long or price_below_ema:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when daily trend turns up OR price breaks above 4H EMA
                if not trend_short or price_above_ema:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_1D_CCI_Trend_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0