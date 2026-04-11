#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly OHLC for Keltner Channel
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ATR for Keltner Channel (20 period)
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_w = pd.Series(tr_w).rolling(window=20, min_periods=20).mean().values
    
    # Weekly EMA for Keltner Channel middle line (20 period)
    ema_w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Upper and lower Keltner Channel bands (2.0 ATR multiplier)
    upper_w = ema_w + 2.0 * atr_w
    lower_w = ema_w - 2.0 * atr_w
    
    # Align weekly Keltner Channel to daily timeframe
    upper_w_daily = align_htf_to_ltf(prices, df_1w, upper_w)
    lower_w_daily = align_htf_to_ltf(prices, df_1w, lower_w)
    
    # Daily ATR for volatility filter (14 period)
    tr1_d = high[1:] - low[1:]
    tr2_d = np.abs(high[1:] - close[:-1])
    tr3_d = np.abs(low[1:] - close[:-1])
    tr_d = np.concatenate([[np.nan], np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))])
    atr_d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume filter: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_w_daily[i]) or np.isnan(lower_w_daily[i]) or
            np.isnan(atr_d[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.8x average)
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Long conditions: price breaks above weekly upper Keltner with volume
        long_signal = volume_confirmed and (price_high > upper_w_daily[i])
        
        # Short conditions: price breaks below weekly lower Keltner with volume
        short_signal = volume_confirmed and (price_low < lower_w_daily[i])
        
        # Exit when price returns to weekly EMA (mean reversion to weekly trend)
        exit_long = position == 1 and price_close < ema_w_daily[i] if 'ema_w_daily' in locals() else False
        exit_short = position == -1 and price_close > ema_w_daily[i] if 'ema_w_daily' in locals() else False
        
        # Align weekly EMA to daily for exit condition
        ema_w_daily = align_htf_to_ltf(prices, df_1w, ema_w)
        exit_long = position == 1 and price_close < ema_w_daily[i]
        exit_short = position == -1 and price_close > ema_w_daily[i]
        
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
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Weekly Keltner Channel breakout strategy for daily timeframe with volume confirmation (>1.8x average volume).
# Enters long when daily price breaks above weekly upper Keltner Band (EMA20 + 2*ATR) with volume >1.8x average.
# Enters short when price breaks below weekly lower Keltner Band (EMA20 - 2*ATR) with same conditions.
# Exits when price returns to the weekly EMA (mean reversion to weekly trend).
# Uses weekly timeframe for trend context and daily for execution to reduce noise.
# Volume filter ensures breakouts have institutional participation.
# Target: 15-25 trades per year to minimize fee drag while capturing strong weekly trends.
# Keltner Channels adapt to volatility, making them effective in both bull and bear markets.