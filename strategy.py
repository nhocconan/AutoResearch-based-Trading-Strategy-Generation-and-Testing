#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter, volume confirmation (>1.5x 20-period average), and ATR(14) stoploss.
# Donchian breakouts capture strong momentum moves; 12h EMA50 ensures alignment with higher timeframe trend to avoid whipsaws.
# Volume confirmation filters for institutional participation; discrete sizing (0.25) minimizes fee churn.
# ATR-based stoploss manages risk during adverse moves. Effective in both bull and bear markets: catches breakouts in trends,
# avoids false breakouts in chop via volume and trend filters. Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.

name = "4h_Donchian20_12hEMA50_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation (on 4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 14)  # 12h EMA50, Donchian, ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_donch_high = donch_high[i]
        curr_donch_low = donch_low[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: price drops below entry_price - 2.0 * atr_at_entry
            # Since we don't track entry_price exactly, use current ATR-based trailing stop
            if curr_close < curr_high - 2.0 * curr_atr:  # trailing stop from session high
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below Donchian low OR trend turns bearish (price below 12h EMA50)
            elif curr_close < curr_donch_low or curr_close < curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: price rises above entry_price + 2.0 * atr_at_entry
            if curr_close > curr_low + 2.0 * curr_atr:  # trailing stop from session low
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above Donchian high OR trend turns bullish (price above 12h EMA50)
            elif curr_close > curr_donch_high or curr_close > curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND above 12h EMA50 AND volume confirmation
            if (curr_close > curr_donch_high and 
                curr_close > curr_ema_12h and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Donchian low AND below 12h EMA50 AND volume confirmation
            elif (curr_close < curr_donch_low and 
                  curr_close < curr_ema_12h and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals