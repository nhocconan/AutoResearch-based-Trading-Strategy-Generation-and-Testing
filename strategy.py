#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume spike (>2.0x 20-period average), and ATR(14) stoploss.
# Donchian breakouts capture strong momentum moves; 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws.
# Volume spike confirms institutional participation; discrete sizing (0.25) minimizes fee churn.
# ATR-based stoploss manages risk without look-ahead.
# Effective in both bull and bear markets: catches breakouts in trends, avoids false breakouts in chop via trend/volume filters.
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.

name = "4h_Donchian20_1dEMA34_VolumeSpike_ATRStop_v1"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channels on 4h timeframe (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = max(34, 20, 14, 20)  # 1d EMA34, Donchian, ATR, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: price drops below entry_price - 2.0 * ATR
            # Trailing exit: price breaks below Donchian Low (10-period for tighter stop)
            donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values[i]
            if curr_low <= entry_price - 2.0 * curr_atr or curr_close < donchian_low_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: price rises above entry_price + 2.0 * ATR
            # Trailing exit: price breaks above Donchian High (10-period for tighter stop)
            donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values[i]
            if curr_high >= entry_price + 2.0 * curr_atr or curr_close > donchian_high_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian High AND above 1d EMA34 AND volume confirmation
            if (curr_high > curr_donchian_high and 
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close  # approximate entry at breakout bar close
            # Short entry: price breaks below Donchian Low AND below 1d EMA34 AND volume confirmation
            elif (curr_low < curr_donchian_low and 
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close  # approximate entry at breakdown bar close
            else:
                signals[i] = 0.0
    
    return signals