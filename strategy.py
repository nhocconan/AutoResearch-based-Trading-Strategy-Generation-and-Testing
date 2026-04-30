#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses discrete sizing 0.25 to limit fee drag. Target: 30-100 total trades over 4 years (7-25/year).
# Donchian channels provide clear structure; breakouts with 1w EMA50 trend filter reduce false signals.
# Volume confirmation ensures institutional participation. Works in both bull and bear markets via trend filter.
# Focus on BTC/ETH with proven edge from Donchian + volume + trend confluence.

name = "1d_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian levels from prior 20 days
    # Upper = max(high[-20:-1]), Lower = min(low[-20:-1])
    prior_high = high[:-1]
    prior_low = low[:-1]
    donchian_upper = pd.Series(prior_high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(prior_low).rolling(window=20, min_periods=20).min().values
    donchian_upper = np.concatenate([[np.full(20, np.nan)], donchian_upper[20:]]) if len(prior_high) >= 20 else np.full(len(close), np.nan)
    donchian_lower = np.concatenate([[np.full(20, np.nan)], donchian_lower[20:]]) if len(prior_low) >= 20 else np.full(len(close), np.nan)
    
    # Calculate 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 20, 50, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Donchian break and 1w EMA50 trend filter
            if curr_volume_spike:
                # Bullish: Close breaks above upper + close above 1w EMA50
                if curr_close > curr_upper and curr_close > curr_ema_50_1w:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below lower + close below 1w EMA50
                elif curr_close < curr_lower and curr_close < curr_ema_50_1w:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR close drops below lower OR loses 1w trend
            if curr_low <= stop_loss or curr_close < curr_lower or curr_close < curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR close rises above upper OR loses 1w trend
            if curr_high >= stop_loss or curr_close > curr_upper or curr_close > curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals