#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter, volume confirmation (>1.5x 20-bar avg),
# and ATR-based stoploss (2.0x ATR). Uses discrete position sizing at ±0.30 to limit fee drag.
# Designed for BTC/ETH: captures strong trends in bull markets and avoids whipsaws in bear markets
# via HTF trend filter and volume confirmation. Target: 20-50 trades/year per symbol to avoid fee drag.

name = "4h_Donchian20_VolumeConfirm_12hEMA50_Trend_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h_vals = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h_vals).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian Channel (20) on 4h
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # ATR (14) for volatility and stoploss
    atr_period = 14
    tr1 = pd.Series(high).rolling(window=2).max().values - pd.Series(low).rolling(window=2).min().values
    tr2 = abs(pd.Series(high).rolling(window=2).max().values - pd.Series(close).shift(1).values)
    tr3 = abs(pd.Series(low).rolling(window=2).min().values - pd.Series(close).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(60, donchian_period, atr_period)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or
            np.isnan(atr[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout above upper channel, price > 12h EMA50, volume confirmation
            if (curr_close > upper_channel[i] and 
                curr_close > curr_ema_50_12h and 
                curr_volume_confirm):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short: Donchian breakout below lower channel, price < 12h EMA50, volume confirmation
            elif (curr_close < lower_channel[i] and 
                  curr_close < curr_ema_50_12h and 
                  curr_volume_confirm):
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: stoploss (2.0x ATR) or mean reversion to middle
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0  # stoploss hit
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit conditions: stoploss (2.0x ATR) or mean reversion to middle
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0  # stoploss hit
                position = 0
            else:
                signals[i] = -0.30
    
    return signals