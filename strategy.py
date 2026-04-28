#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze Breakout with 12h EMA50 trend filter and volume confirmation
# Bollinger Band squeeze identifies low volatility periods preceding breakouts. 12h EMA50 ensures higher-timeframe trend alignment.
# Volume spike (>2.0x 20-bar average) confirms breakout validity. Target 19-50 trades/year to minimize fee drag.
# Works in bull/bear markets by capturing explosive moves after consolidation, with volume confirmation filtering false breakouts.

name = "4h_BollingerSqueeze_12hEMA50_Trend_VolumeSpike_v1"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Bollinger Bands (4h)
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + bb_std * std_dev
    lower_band = sma - bb_std * std_dev
    bb_width = (upper_band - lower_band) / sma  # Normalized bandwidth
    
    # Bollinger Band Squeeze: bandwidth below 20-period average bandwidth
    bb_width_series = pd.Series(bb_width)
    bb_width_ma = bb_width_series.rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma  # True when in squeeze (low volatility)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(bb_period, 20, 50)  # BB period, volume MA, EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(sma[i]) or np.isnan(std_dev[i]) or 
            np.isnan(bb_width_ma[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        is_squeeze = squeeze[i]
        upper = upper_band[i]
        lower = lower_band[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: breakout above upper band after squeeze, price above 12h EMA50, volume spike
            if price > upper and is_squeeze and price > ema_50_12h_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: breakout below lower band after squeeze, price below 12h EMA50, volume spike
            elif price < lower and is_squeeze and price < ema_50_12h_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or mean reversion to middle band
            # ATR-based stoploss: 2.0 * ATR below entry (using 4h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.0 * atr_val
            # Exit on stoploss or when price reverts to middle band (mean reversion)
            if price < stop_loss or price <= sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or mean reversion to middle band
            # ATR-based stoploss: 2.0 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.0 * atr_val
            # Exit on stoploss or when price reverts to middle band (mean reversion)
            if price > stop_loss or price >= sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals