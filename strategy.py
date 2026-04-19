# 1h_4d_1d_MomentumBreakout_v1
# Hypothesis: 1h momentum breakouts with 4h trend filter and 1d volume filter work in both bull/bear markets by capturing strong moves while avoiding chop.
# Uses 4h EMA50 for trend direction, 1h price action for entry timing, and 1d volume surge for confirmation.
# Target: 15-30 trades/year by requiring 4h trend alignment + 1h breakout + 1d volume spike.
# Timeframe: 1h, Leverage: 1.0

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_1d_MomentumBreakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d average volume for volume filter
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 1h indicators
    # ATR for stop calculation (not used in signal but for volatility context)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1h price change for momentum
    price_change = (close - np.roll(close, 1)) / np.roll(close, 1)
    price_change[0] = 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        atr_val = atr[i]
        mom = price_change[i]
        
        # Volume filter: current 1h volume > 2.0x 20-day average volume (scaled to hourly)
        # Approximate: 1d volume / 24 = avg hourly volume
        vol_filter = vol > 2.0 * (vol_ma / 24.0)
        
        # Trend filter: price > 4h EMA50 for long, < for short
        long_trend = price > ema50_4h_aligned[i]
        short_trend = price < ema50_4h_aligned[i]
        
        # Momentum filter: significant price move
        mom_filter = abs(mom) > 0.008  # 0.8% minimum move
        
        if position == 0:
            # Long: bullish momentum + uptrend + volume surge
            if mom > 0 and mom_filter and long_trend and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: bearish momentum + downtrend + volume surge
            elif mom < 0 and mom_filter and short_trend and vol_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: momentum fails or trend reverses
            if mom < -0.004 or not long_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: momentum fails or trend reverses
            if mom > 0.004 or not short_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals