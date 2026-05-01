#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND volume > 1.5x 20-period average AND price > 1w EMA34.
# Short when price breaks below Donchian(20) low AND volume > 1.5x 20-period average AND price < 1w EMA34.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend).
# Target: 20-40 trades/year on 1d timeframe.

name = "1d_Donchian_20_Breakout_1wEMA34_Volume_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 1d data ONCE before loop for Donchian and volume filters (primary timeframe data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian(20) on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Load 1w data ONCE before loop for EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Donchian, volume, and EMA
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike: current volume > 1.5x 1d volume average
        if vol_ma_1d_aligned[i] <= 0 or np.isnan(vol_ma_1d_aligned[i]):
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma_1d_aligned[i] * 1.5)
        
        # Donchian breakout
        breakout_long = curr_high > donch_high_aligned[i]
        breakout_short = curr_low < donch_low_aligned[i]
        
        # Trend filter: price vs 1w EMA34
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout AND volume spike AND uptrend
            if breakout_long and volume_spike and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Donchian breakdown AND volume spike AND downtrend
            elif breakout_short and volume_spike and downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low OR trend turns down
            elif curr_close < donch_low_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high OR trend turns up
            elif curr_close > donch_high_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals