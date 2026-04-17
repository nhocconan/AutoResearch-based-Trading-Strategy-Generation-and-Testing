# 4h Donchian Breakout + 12h EMA Trend + Volume Confirmation + ATR Stoploss
# Long when price breaks above Donchian(20) high + 12h EMA34 rising + volume > 1.5x 4h volume SMA(20)
# Short when price breaks below Donchian(20) low + 12h EMA34 falling + volume > 1.5x 4h volume SMA(20)
# Exit when price returns to Donchian midpoint or EMA direction flips
# Uses volume confirmation and EMA trend filter to reduce false breakouts
# Designed for 4h timeframe with 12h trend filter for multi-timeframe confluence
# Target: 20-50 trades/year per symbol to avoid fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(34) on 12h
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h volume SMA(20)
    vol_sma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for dynamic stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = max(20, 34)  # need Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_sma_4h[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_4h[i]
        ema_val = ema_12h_aligned[i]
        ema_prev = ema_12h_aligned[i-1] if i > 0 else ema_val
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above Donchian high + EMA rising + volume spike
            if price > donch_high[i-1] and ema_val > ema_prev and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below Donchian low + EMA falling + volume spike
            elif price < donch_low[i-1] and ema_val < ema_prev and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price returns to midpoint OR EMA flips down OR stoploss hit
            if price < donch_mid[i] or ema_val < ema_prev or price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to midpoint OR EMA flips up OR stoploss hit
            if price > donch_mid[i] or ema_val > ema_prev or price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_12hEMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0