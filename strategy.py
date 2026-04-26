#!/usr/bin/env python3
"""
1h_Contrarian_RSI_With_Volume_Spike_And_HTF_Trend_Filter
Hypothesis: On 1h timeframe, enter contrarian RSI mean-reversion trades (RSI<30 long, RSI>70 short) only when aligned with 4h trend (EMA50) and confirmed by volume spike (>1.5x 20-bar median). Exit on RSI mean-reversion (RSI>50 for longs, RSI<50 for shorts) or ATR-based stoploss (2x ATR). Uses 1h for entry timing, 4h for trend filter. Designed to work in both bull and bear markets via trend filter and mean-reversion logic. Targets 15-35 trades/year (~60-140 over 4 years) by requiring confluence of RSI extreme, volume spike, and HTF trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h ATR(14) for stoploss
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Reduced fixed position size to control trade frequency and drawdown
    fixed_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for RSI stability, 20 for volume median, 50 for 4h EMA/ATR
    start_idx = max(34, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(atr_14_4h_aligned[i]) or
            np.isnan(rsi_values[i]) or
            np.isnan(vol_median_20[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_4h_aligned[i]
        atr_val = atr_14_4h_aligned[i]
        vol_spike = volume_spike[i]
        rsi_val = rsi_values[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: RSI < 30 (oversold), volume spike, and uptrend (close > 4h EMA50)
            long_entry = (rsi_val < 30) and vol_spike and (close_val > ema_50_val)
            # Short: RSI > 70 (overbought), volume spike, and downtrend (close < 4h EMA50)
            short_entry = (rsi_val > 70) and vol_spike and (close_val < ema_50_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on RSI mean-reversion (RSI>50), ATR stoploss, or trend reversal
            stop_price = entry_price - 2.0 * atr_val
            if (rsi_val > 50) or (close_val < stop_price) or (close_val < ema_50_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on RSI mean-reversion (RSI<50), ATR stoploss, or trend reversal
            stop_price = entry_price + 2.0 * atr_val
            if (rsi_val < 50) or (close_val > stop_price) or (close_val > ema_50_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Contrarian_RSI_With_Volume_Spike_And_HTF_Trend_Filter"
timeframe = "1h"
leverage = 1.0