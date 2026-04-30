#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses discrete sizing 0.30 to balance profit potential and drawdown control.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull markets (breakouts continue trend)
# and bear markets (breakdowns continue downtrend). ATR-based stoploss controls drawdown.

name = "4h_Donchian20_1dEMA34_Volume_ATR_v2"
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
    
    # Calculate 1d Donchian(20) from prior 1d bar (to avoid look-ahead)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # Need 20 periods + 1 for prior
        return np.zeros(n)
    
    # Prior 20-day high/low for Donchian channels (shifted to avoid look-ahead)
    prior_high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    prior_low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 4h timeframe (wait for 1d bar to close)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, prior_high_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, prior_low_20)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.8x 30-period average (stricter to reduce trades)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma_30)
    
    # ATR for stoploss (15-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_15 = pd.Series(tr).rolling(window=15, min_periods=15).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 30, 34, 15)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_30[i]) or
            np.isnan(atr_15[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper_20_aligned[i]
        curr_lower = lower_20_aligned[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_15[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Donchian break and 1d trend filter
            if curr_volume_spike:
                # Bullish: Close breaks above upper Donchian + price above 1d EMA34
                if curr_close > curr_upper and curr_close > curr_ema_34_1d:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below lower Donchian + price below 1d EMA34
                elif curr_close < curr_lower and curr_close < curr_ema_34_1d:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry (wider for 4h volatility)
            stop_loss = entry_price - 2.5 * curr_atr
            # Exit: Stoploss hit OR close drops below upper Donchian OR loses 1d trend
            if curr_low <= stop_loss or curr_close < curr_upper or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry
            stop_loss = entry_price + 2.5 * curr_atr
            # Exit: Stoploss hit OR close rises above lower Donchian OR loses 1d trend
            if curr_high >= stop_loss or curr_close > curr_lower or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals