#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high with volume > 1.3x 20-period volume average and price > 1w EMA34.
# Short when price breaks below 20-period Donchian low with volume confirmation and price < 1w EMA34.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Donchian channels calculated from prior completed 1d bar to avoid look-ahead.
# Volume confirmation filters low-momentum breakouts. 1w EMA34 ensures trades only in established trends.
# Works in bull (breakouts with strong uptrend) and bear (breakouts with strong downtrend) regimes.
# Target: 15-25 trades/year on 1d timeframe.

name = "1d_Donchian_20_Breakout_1wEMA34_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Load 1w data ONCE before loop for EMA and volume filters (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1w volume average (20-period)
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Load 1d data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Pre-calculate Donchian channels for all 1d bars (using previous completed bar)
    # Upper channel = max(high of last 20 bars)
    # Lower channel = min(low of last 20 bars)
    # We'll calculate for each position i using the previous 20 completed 1d bars
    upper_channel = np.full(len(high_1d), np.nan)
    lower_channel = np.full(len(low_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        # Use previous 20 completed bars (i-20 to i-1)
        upper_channel[i] = np.max(high_1d[i-20:i])
        lower_channel[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian channels to 1d timeframe (no shift needed as we used completed bars)
    # Since we're on 1d timeframe, the aligned array is the same as calculated
    upper_channel_aligned = upper_channel
    lower_channel_aligned = lower_channel
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, and Donchian
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_1w_aligned[i]) or
            np.isnan(upper_channel_aligned[i]) or
            np.isnan(lower_channel_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.3x 1w volume average
        if vol_ma_1w_aligned[i] <= 0 or np.isnan(vol_ma_1w_aligned[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma_1w_aligned[i] * 1.3)
        
        # Trend filter: price vs 1w EMA34
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND volume confirmation AND uptrend
            if (curr_high > upper_channel_aligned[i] and 
                volume_confirm and 
                uptrend):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Donchian breakout down AND volume confirmation AND downtrend
            elif (curr_low < lower_channel_aligned[i] and 
                  volume_confirm and 
                  downtrend):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channels OR trend reverses
            elif (curr_low >= lower_channel_aligned[i] and curr_low <= upper_channel_aligned[i]) or \
                 (curr_close < ema_34_1w_aligned[i]):  # trend reversal
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channels OR trend reverses
            elif (curr_high >= lower_channel_aligned[i] and curr_high <= upper_channel_aligned[i]) or \
                 (curr_close > ema_34_1w_aligned[i]):  # trend reversal
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals