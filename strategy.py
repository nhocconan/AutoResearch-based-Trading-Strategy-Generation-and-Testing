#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly high, low, close for pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot and ranges
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly Camarilla levels
    r3_1w = pivot_1w + (range_1w * 1.1 / 4)
    r4_1w = pivot_1w + (range_1w * 1.1 / 2)
    s3_1w = pivot_1w - (range_1w * 1.1 / 4)
    s4_1w = pivot_1w - (range_1w * 1.1 / 2)
    
    # Shift by 1 to use only completed weekly bars
    r3_1w = np.roll(r3_1w, 1)
    r4_1w = np.roll(r4_1w, 1)
    s3_1w = np.roll(s3_1w, 1)
    s4_1w = np.roll(s4_1w, 1)
    r3_1w[0] = np.nan
    r4_1w[0] = np.nan
    s3_1w[0] = np.nan
    s4_1w[0] = np.nan
    
    # Align weekly levels to daily timeframe
    r3_1d = align_htf_to_ltf(prices, df_1w, r3_1w)
    r4_1d = align_htf_to_ltf(prices, df_1w, r4_1w)
    s3_1d = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1d = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Daily ATR for volatility filter and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter: close > 50 EMA for long, < 50 EMA for short
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1d[i]) or np.isnan(r4_1d[i]) or np.isnan(s3_1d[i]) or np.isnan(s4_1d[i]) or
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
        long_signal = volume_confirmed and vol_filter and (price_low < s3_1d[i] or price_low < s4_1d[i]) and (price_close > ema_val)
        
        # Short conditions: price breaks above R3 or R4 (overbought) with volume, vol filter, and below EMA50
        short_signal = volume_confirmed and vol_filter and (price_high > r3_1d[i] or price_high > r4_1d[i]) and (price_close < ema_val)
        
        # Exit when price returns to weekly pivot level
        pivot_1d = align_htf_to_ltf(prices, df_1w, pivot_1w)
        exit_long = position == 1 and price_close > pivot_1d[i]
        exit_short = position == -1 and price_close < pivot_1d[i]
        
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

# Hypothesis: Weekly Camarilla levels act as strong support/resistance for daily price action.
# Enters long when daily price breaks below S3/S4 (oversold bounce) with volume confirmation (>1.5x average),
# sufficient volatility (ATR > 0.5% of price), and above daily 50 EMA (trend filter).
# Enters short when price breaks above R3/R4 (overbought rejection) with same conditions plus below EMA50.
# Exits when price returns to weekly pivot level, capturing mean reversion.
# Trend filter reduces whipsaws in strong trends. Volume and volatility filters reduce false breaks.
# Designed for 8-20 trades per month (~100-240/year) to balance opportunity and fee cost on daily timeframe.
# Works in both bull (buying dips) and bear (selling rallies) markets.