#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses discrete sizing 0.25 to balance capture and fee drag. Target: 75-150 total trades over 4 years (19-37/year).
# Donchian provides clear structural breakouts; 1w EMA50 filters for primary trend alignment.
# Volume spike ensures institutional participation. Works in bull via breakouts with trend, in bear via mean-reversion failsafes.

name = "12h_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to reduce noise
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_m = (donchian_h + donchian_l) / 2  # Middle for exit logic
    
    # Calculate 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0x 28-period average (strict to reduce trades)
    vol_ma_28 = pd.Series(volume).rolling(window=28, min_periods=28).mean().values
    volume_spike = volume > (2.0 * vol_ma_28)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 28, 50, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_28[i]) or
            np.isnan(atr_14[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_h = donchian_h[i]
        curr_donchian_l = donchian_l[i]
        curr_donchian_m = donchian_m[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Donchian break and 1w EMA50 trend filter
            if curr_volume_spike:
                # Bullish: Close breaks above upper Donchian + close above 1w EMA50
                if curr_close > curr_donchian_h and curr_close > curr_ema_50_1w:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below lower Donchian + close below 1w EMA50
                elif curr_close < curr_donchian_l and curr_close < curr_ema_50_1w:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry (wider for 12h volatility)
            stop_loss = entry_price - 2.5 * curr_atr
            # Exit: Stoploss hit OR close drops below middle Donchian OR loses 1w trend
            if curr_low <= stop_loss or curr_close < curr_donchian_m or curr_close < curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry
            stop_loss = entry_price + 2.5 * curr_atr
            # Exit: Stoploss hit OR close rises above middle Donchian OR loses 1w trend
            if curr_high >= stop_loss or curr_close > curr_donchian_m or curr_close > curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals