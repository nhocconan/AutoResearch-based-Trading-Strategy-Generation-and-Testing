#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_trend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3, R4, S3, S4
    r3 = pivot + (range_1d * 1.1 / 4)
    r4 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 4)
    s4 = pivot - (range_1d * 1.1 / 2)
    
    # Shift by 1 to use only completed daily bars
    r3 = np.roll(r3, 1)
    r4 = np.roll(r4, 1)
    s3 = np.roll(s3, 1)
    s4 = np.roll(s4, 1)
    r3[0] = np.nan
    r4[0] = np.nan
    s3[0] = np.nan
    s4[0] = np.nan
    
    # Align daily levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 4h ATR for volatility filter and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: 4h close > 50 EMA for long, < 50 EMA for short
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r3_4h[i]) or np.isnan(r4_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(s4_4h[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        ema_val = ema_50[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_val > 0.005 * price_close  # ATR > 0.5% of price
        
        # Long conditions: price breaks below S3 or S4 (oversold) with volume, vol filter, and above EMA50
        long_signal = volume_confirmed and vol_filter and (price_low < s3_4h[i] or price_low < s4_4h[i]) and (price_close > ema_val)
        
        # Short conditions: price breaks above R3 or R4 (overbought) with volume, vol filter, and below EMA50
        short_signal = volume_confirmed and vol_filter and (price_high > r3_4h[i] or price_high > r4_4h[i]) and (price_close < ema_val)
        
        # Exit when price returns to daily pivot level
        pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
        exit_long = position == 1 and price_close > pivot_4h[i]
        exit_short = position == -1 and price_close < pivot_4h[i]
        
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

# Hypothesis: Daily Camarilla levels act as strong support/resistance for 4h price action.
# Enters long when 4h price breaks below S3/S4 (oversold bounce) with volume confirmation (>1.5x average),
# sufficient volatility (ATR > 0.5% of price), and above 4h 50 EMA (trend filter).
# Enters short when price breaks above R3/R4 (overbought rejection) with same conditions plus below EMA50.
# Exits when price returns to daily pivot level, capturing mean reversion.
# Trend filter reduces whipsaws in strong trends. Volume and volatility filters reduce false breaks.
# Designed for 4-8 trades per month (~50-100/year) to minimize fee drag on 4h timeframe.
# Works in both bull (buying dips) and bear (selling rallies) markets.